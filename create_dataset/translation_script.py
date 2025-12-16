#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from google.cloud import translate
import pandas as pd
import random
import numpy as np

# ============================================================= #
# ================== PATH HANDLING ============================= #
# ============================================================= #

# load dataset for translation 
DATA_DIR = Path(os.environ.get("DATA_DIR"))
print("loading data from:", DATA_DIR)

# where to save your translated data 
OUT_DIR = Path(os.environ.get("RAW_TRANSLATIONS_DIR", "./raw_translations"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("saving data to:", OUT_DIR)

# ============================================================= #
# ================== CONFIGURATION ============================= #
# ============================================================= #

JSON_KEY = os.environ.get("JSON_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 1)) 

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

gest_df = pd.read_csv(DATA_DIR)

if SAMPLE_SIZE != 1:
    # if testing with a few examples 
    TEST_GENDERED_LANG = random.choice(list(GENDERED_LANGS.keys()))
    TEST_NEUTRAL_LANG = random.choice(list(NEUTRAL_LANGS.keys()))
    test_langs = [TEST_GENDERED_LANG, TEST_NEUTRAL_LANG]
    sampled_df = gest_df.sample(n=SAMPLE_SIZE)
else:
    # Full run
    test_langs = list(ALL_LANGS.keys())
    sampled_df = gest_df

print("Testing languages: ", test_langs)
print("Number of sentences to translate: ", len(sampled_df))
print("Sample from dataset: ")
print(gest_df.head(2))

# ============================================================= #
# ================== GOOGLE TRANSLATE CLIENT=================== #
# ============================================================= #

client = translate.TranslationServiceClient.from_service_account_json(
    str(JSON_KEY)
)

def translate_text(text, target_language_code):
    location = "global"
    parent = f"projects/{PROJECT_ID}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "en",
            "target_language_code": target_language_code,
        }
    )

    return response.translations[0].translated_text

# ============================================================= #
# ================== MAIN LOOP ================================= #
# ============================================================= #

# iterate over the languages
for language in test_langs:
    code = ALL_LANGS[language]
    is_gendered = language in GENDERED_LANGS

    save_path = OUT_DIR / f"{language.lower().replace(' ', '_')}.csv"

    if is_gendered:
        columns = ["GEST_sentence", "the man said", "the woman said", "original_stereotype"]
    else:
        columns = ["GEST_sentence", "translation", "original_stereotype"]

    # We use the index of the sampled dataframe for the output dataframe
    df_out = pd.DataFrame(index=sampled_df.index, columns=columns)
    df_out["GEST_sentence"] = sampled_df["sentence"]
    df_out["original_stereotype"] = sampled_df["stereotype"] # Keep track of the original stereotype

    print(f"Translating {len(sampled_df)} sentences for language: {language}")

    # Iterate over the sampled sentences
    for idx, row in sampled_df.iterrows():
        sentence = row["sentence"]

        if is_gendered:
            df_out.at[idx, "the man said"] = translate_text(f"The man said, '{sentence}'", code)
            df_out.at[idx, "the woman said"] = translate_text(f"The woman said, '{sentence}'", code)
        else:
            df_out.at[idx, "translation"] = translate_text(sentence, code)

        # Save after each sentence (££ safety)
        df_out.to_csv(save_path)

    print(f"Saved translations for {language}")
