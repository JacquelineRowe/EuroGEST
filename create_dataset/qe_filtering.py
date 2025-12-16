#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
from pathlib import Path
import pandas as pd
from comet import download_model, load_from_checkpoint
import torch
import numpy as np


# ======================================================== #
# ===================== DEVICE SETUP ===================== #
# ======================================================== #

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple Metal GPU acceleration
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA GPU if available
else:
    DEVICE = torch.device("cpu") 

print(f"Using device: {DEVICE}")

# ============================================================= #
# ================== PATH HANDLING ============================ #
# ============================================================= #

# load data
DATA_DIR = Path(os.environ.get("RAW_TRANSLATIONS_DIR", "./raw_translations"))
print("loading translated data from:", DATA_DIR)

# where to save your filtered data 
OUT_DIR = Path(os.environ.get("FILTERED_TRANSLATIONS_DIR", "./filtered_translations"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("saving data to:", OUT_DIR)

THRESHOLD = float(os.environ.get("THRESHOLD", "0.86"))
QE_MODEL = os.environ.get("QE_MODEL", "Unbabel/wmt22-cometkiwi-da")


# ============================================================= #
# ================== CONFIGURATION ============================ #
# ============================================================= #

GENDERED_LANGS = {
    'Bulgarian': 'bg', 'French': 'fr', 'German': 'de', 'Greek': 'el',
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Spanish': 'es', 'Catalan': 'ca',
    'Galician': 'gl', 'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Russian': 'ru', 'Ukrainian': 'uk'
}

NEUTRAL_LANGS = {
    'Danish': 'da', 'Dutch': 'nl', 'Estonian': 'et', 'Finnish': 'fi',
    'Hungarian': 'hu', 'Irish': 'ga', 'Swedish': 'sv',
    'Norwegian': 'no', 'Turkish': 'tr'
}

ALL_LANGS = {**GENDERED_LANGS, **NEUTRAL_LANGS}
STEREOTYPES = range(1, 17) 

model_path = download_model(QE_MODEL)
qe_model = load_from_checkpoint(model_path)
qe_model.eval()

# ===================== UTILITY FUNCTIONS ===================== #
def score_translations(sources, targets, model=qe_model, batch_size=16, gpus=1):
    samples = [{"src": s, "mt": t} for s, t in zip(sources, targets)]
    outputs = model.predict(
        samples=samples,
        batch_size=batch_size,
        gpus=gpus,
        accelerator="auto"
    )
    system_score, segments_scores = outputs.system_score, outputs.scores
    torch.cuda.empty_cache()
    return system_score, segments_scores

def highlight_bad_translations(sentence_scores, sources, targets, threshold=THRESHOLD):
    indexes_to_remove = []
    for i, score in enumerate(sentence_scores):
        if score <= threshold:
            print(f"Bad translation (score={score:.2f}):")
            print(f"Source: {sources[i]}")
            print(f"Target: {targets[i]}")
            indexes_to_remove.append(i)
    return indexes_to_remove

# ===================== MAIN LOOP ===================== #
# track some statistics about overall quality and number of sentences removed in each lang

system_scores_neutral = {}
system_scores_gendered = {}
sentence_scores_neutral = {}
sentence_scores_gendered = {}
coverage = {}

# sentence_scores_neutral[lang] = sent_scores
for language in list(ALL_LANGS.keys()):
    print(f"Processing language {language}")

    # Load raw translations
    raw_translations_file = DATA_DIR / f"{language.lower().replace(' ', '_')}.csv"

    if os.path.exists(raw_translations_file):
        raw_translations = pd.read_csv(raw_translations_file, index_col=0)
        print("Sample of raw translations: ", raw_translations.head(2))
    else:
        print(f"File {raw_translations_file} does not exist. Skipping language {language}.")
        continue    
    
    if language in NEUTRAL_LANGS:
        src_sentences = list(raw_translations["GEST_sentence"])
        tgt_sentences = list(raw_translations["translation"])
        sys_scores, sent_scores = score_translations(src_sentences, tgt_sentences)
        idxs_to_remove = highlight_bad_translations(sent_scores, src_sentences, tgt_sentences)

        system_scores_neutral[language] = sys_scores
        sentence_scores_neutral[language] = sent_scores

    elif language in GENDERED_LANGS:
        original_sentences = list(raw_translations["GEST_sentence"])
        src_sentences_masc = [f"The man said, '{sentence}'" for sentence in original_sentences]
        src_sentences_fem = [f"The woman said, '{sentence}'" for sentence in original_sentences]
        tgt_sentences_masc = list(raw_translations["the man said"])
        tgt_sentences_fem = list(raw_translations["the woman said"])   

        # Score male
        sys_score_masc, sent_scores_masc = score_translations(src_sentences_masc, tgt_sentences_masc)
        idxs_to_remove_masc = highlight_bad_translations(sent_scores_masc, src_sentences_masc, tgt_sentences_masc)

        # Score female
        sys_score_fem, sent_scores_fem = score_translations(src_sentences_fem, tgt_sentences_fem)
        idxs_to_remove_fem = highlight_bad_translations(sent_scores_fem, src_sentences_fem, tgt_sentences_fem)
        # Combine indexes to remove

        idxs_to_remove = set(idxs_to_remove_masc).union(set(idxs_to_remove_fem))            

        system_scores_gendered[language] = (sys_score_masc, sys_score_fem, (sys_score_masc + sys_score_fem) / 2)
        sentence_scores_gendered[language] = (sent_scores_masc, sent_scores_fem, (np.array(sent_scores_masc) + np.array(sent_scores_fem)) / 2)

    # convert to original GEST id indices
    if len(idxs_to_remove) > 0:
        idxs_to_drop = raw_translations.index[idxs_to_remove]
        filtered_df = raw_translations.drop(list(idxs_to_drop))
    else:   
        filtered_df = raw_translations

    # save filtered data 
    filtered_df.to_csv(OUT_DIR / f"{language.lower().replace(' ', '_')}_filtered.csv")
    coverage[language] = len(filtered_df)/len(raw_translations)

# save system scores 
system_scores_neutral_df = pd.Series(system_scores_neutral, name="Avg All Score").to_frame()
system_scores_gendered_df = pd.DataFrame(system_scores_gendered).T
system_scores_gendered_df.columns=["Avg Masc Score", "Avg Fem Score", "Avg All Score"]
system_scores_df = pd.concat([system_scores_neutral_df, system_scores_gendered_df], axis=1)
system_scores_df = system_scores_df.drop(columns=["Avg All Score"])
unified_avg_score = system_scores_gendered_df["Avg All Score"].combine_first(
    system_scores_neutral_df["Avg All Score"]
)
system_scores_df['Avg All Score'] = unified_avg_score

coverage_df = pd.Series(coverage, name="Coverage").to_frame()

system_scores_df.to_csv(OUT_DIR / "system_scores.csv")
coverage_df.to_csv(OUT_DIR / "coverage.csv")

