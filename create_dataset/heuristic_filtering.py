#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from pathlib import Path
import os
import numpy as np

# ============================================================= #
# ================== PATH HANDLING ============================ #
# ============================================================= #

# load data
GEST_DIR = Path(os.environ.get("DATA_DIR"))
DATA_DIR = Path(os.environ.get("FILTERED_TRANSLATIONS_DIR", "./filtered_translations"))
print("loading filtered data from:", DATA_DIR)

# where to save your final data 
OUT_DIR = Path(os.environ.get("FINAL_DATA_DIR", "./final_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("saving data to:", OUT_DIR)

NUM_GENDERED_WORDS = int(os.environ.get("NUM_GENDERED_WORDS", 1))
NUM_DIFFERENT_LETTERS = int(os.environ.get("NUM_DIFFERENT_LETTERS", 2))


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

# ============================================================== #
# ========= UTILITY FUNCTIONS=================================== #
# ============================================================== #

def extract_quoted_sentence(text):
    ''' function to extract the stereotype sentence from quotation marks in the
    gender-forced translated sentence '''
    text = text.replace("‘", "'").replace("’", "'")
    quoted_sentence = re.search(r'«([^"]+)»', text)
    if quoted_sentence == None: 
        quoted_sentence = re.search(r'"([^"]+)"', text)
        if quoted_sentence == None:
            quoted_sentence = re.search(r"'([^']+)'", text)
            ## deal with exceptions where ' is used mid phrase
            if quoted_sentence and len(quoted_sentence[1]) <= 3:
                quoted_sentence = re.search(r"'(.*)'", text)
            if quoted_sentence == None:
                quoted_sentence = re.search(r'「([^"]+)」', text)
                if quoted_sentence == None: 
                    quoted_sentence = re.search(r'„([^"]+)(“|”)', text)
                    if quoted_sentence == None:
                        quoted_sentence = re.search(r'“([^”]+)', text)
                        if quoted_sentence == None:
                            quoted_sentence = re.search("‚([^’]+)", text)
                            if quoted_sentence == None:
                                quoted_sentence = re.search(",([^']+)", text)
                                if quoted_sentence == None:
                                    print("Error in extracting quoted sentence (missing opening or closing quote marks)")
                                    print(text) 
                                    quoted_sentence = [f'"ERROR": {text}']
    if quoted_sentence == None:
        print("Error in extracting quoted sentence")
        print(text) 
        quoted_sentence = [f'"ERROR": {text}']
        print(f"Error in extracting quoted sentence from: {text}", flush=True)
        return "ERROR"
    else:
        quoted_sentence = quoted_sentence[0].strip()
        # strip punc
        pattern = r'^[^\w\s]+|[^\w\s]+$'
        quoted_sentence = re.sub(pattern, '', quoted_sentence) 
        return quoted_sentence


def compare_translations(masculine_translation, feminine_translation):
    gendered_words = []
    if pd.isna(masculine_translation) or pd.isna(feminine_translation):   
        return "Unknown", None
    else:
        # remove any punctuation remaining inside the sentence 
        pattern = r'[^\w\s]'
        masculine_translation = re.sub(pattern, '', masculine_translation, flags=re.UNICODE).strip() 
        feminine_translation = re.sub(pattern, '', feminine_translation, flags=re.UNICODE).strip() 
        pattern = r'  '
        masculine_translation = re.sub(pattern, ' ', masculine_translation, flags=re.UNICODE) 
        feminine_translation = re.sub(pattern, ' ', feminine_translation, flags=re.UNICODE) 
        if masculine_translation == feminine_translation:
            return "Neutral", None
        else:
            ## check if character length difference is greater than 5
            if abs(len(masculine_translation) - len(feminine_translation)) > 5:
                return "Unknown", None
            else:
                # check if transaltions differ by more than 2 words 
                masculine_words = masculine_translation.split()
                feminine_words = feminine_translation.split()
                if len(masculine_words) != len(feminine_words):
                    return "Unknown", None
                else:
                    count_different_words = 0
                    different_word_indexes = []
                    for word_index, (masculine_word, feminine_word) in enumerate(zip(masculine_words, feminine_words)):
                        if masculine_word != feminine_word:
                            count_different_words += 1
                            different_word_indexes.append(word_index)
                    if count_different_words > NUM_GENDERED_WORDS:
                        return "Unknown", None
                    else:
                        if len(different_word_indexes) == 0:
                            print(masculine_translation)
                            print(feminine_translation)
                        else:
                            # check if different words differ by only two lertters 
                            for different_word_index in different_word_indexes:
                                masculine_word = masculine_words[different_word_index]
                                feminine_word = feminine_words[different_word_index]
                                masc_length = len(masculine_word)
                                fem_length = len(feminine_word)
                                longest_word_length = min(masc_length, fem_length)
                                num_different_letters_length = np.abs(masc_length-fem_length)
                                max_different_letters = NUM_DIFFERENT_LETTERS - num_different_letters_length
                                
                                count_different_letters = 0
                                for letter_index in range(0, longest_word_length):
                                    if masculine_word[letter_index].casefold() == feminine_word[letter_index].casefold():
                                        pass
                                    else:
                                        count_different_letters += 1
                                if count_different_letters <= max_different_letters:
                                    gendered_words.append((masculine_word, feminine_word))
                                else:                                            
                                    return "Unknown", None
            if len(gendered_words) >= 1:
                return "Gendered", gendered_words
            


def count_non_null_rows(df, column):
    total = 0
    sentences = df[column]
    for row in sentences:
        if not pd.isna(row):
            total += 1
    return total

def count_stereotype_samples(language_data: pd.DataFrame, stereotypes: list) -> dict:
    """
    Filters a DataFrame to include only rows with at least one translation 
    (neutral, masculine, or feminine) and then counts the number of samples 
    for each stereotype based on the filtered data.
    """

    translation_cols = [
        "neutral translation", 
        "masculine translation"
    ]

    all_missing_mask = language_data[translation_cols].isna().all(axis=1)
    at_least_one_translation_mask = ~all_missing_mask
    translated_data = language_data[at_least_one_translation_mask].copy()

    num_samples_per_stereotype = {}
    
    for stereotype in STEREOTYPES:
        num_samples = len(translated_data[translated_data["original_stereotype"] == stereotype])
        num_samples_per_stereotype[stereotype] = num_samples
        
    return num_samples_per_stereotype


# ============================================================== #
# =============== MAIN LOOP ==================================== #
# ============================================================== #
gest_df = pd.read_csv(GEST_DIR)

# initialise dfs to store key stats about each language's final data
stats_to_collect = ["Neutral",
                    "Gendered",
                    "Discarded during heuristic filtering",
                    "Discarded during QE filtering",
                    f"Coverage",
                    "Number samples per stereotype"]

stats_df = pd.DataFrame(index=ALL_LANGS.keys(), columns=stats_to_collect)

for language in list(ALL_LANGS.keys()):

    language_data = pd.DataFrame(columns=["GEST sentence", 
                                 "neutral translation", 
                                 "masculine translation", 
                                 "feminine translation", 
                                 "gendered word masculine",
                                 "gendered word feminine",
                                 "original_stereotype"])
    
    language_data["GEST sentence"] = gest_df["sentence"]
    language_data["original_stereotype"] = gest_df["stereotype"]

    print(f"Processing language {language}")

    filtered_data_path = DATA_DIR / f"{language.lower().replace(' ', '_')}_filtered.csv"

    if language in NEUTRAL_LANGS:

        if os.path.exists(filtered_data_path):
            filtered_data = pd.read_csv(filtered_data_path, index_col=0)
            # check how many were discarded during QE filtering and store for results overview 
            filtered_data_clean = filtered_data.dropna(subset=["translation"], 
                how='any' 
            )
            num_discarded_qe_filtering = len(gest_df) - len(filtered_data_clean)
            stats_df.at[language, "Discarded during QE filtering"] = num_discarded_qe_filtering
            print(num_discarded_qe_filtering)
        else:
            print(f"File {filtered_data_path} does not exist. Skipping language {language}.")
            continue    
        # no filtering needed, just use QE data
        for index, gest_row in filtered_data_clean.iterrows():
            language_data.at[index, "neutral translation"] = gest_row["translation"]
            num_neutral_sentences = count_non_null_rows(language_data, "neutral translation")

        num_samples_per_stereotype = count_stereotype_samples(language_data, STEREOTYPES)

        stats_df.at[language, "Neutral"] = num_neutral_sentences
        stats_df.at[language, "Gendered"] = 0
        stats_df.at[language, "Discarded during heuristic filtering"] = 0
        stats_df.at[language, "Coverage"] = f"{(num_neutral_sentences/len(language_data))*100:.2f}"
        stats_df.at[language, "Number samples per stereotype"] = str(num_samples_per_stereotype)

        language_data = language_data.drop(columns=["masculine translation", "feminine translation"])
        
    elif language in GENDERED_LANGS:

        if os.path.exists(filtered_data_path):
            filtered_data = pd.read_csv(filtered_data_path, index_col=0)
            # check how many were discarded during QE filtering and store for results overview 
            filtered_data_clean = filtered_data.dropna(subset=["the man said", "the woman said"], 
                how='any' 
            )
            num_discarded_qe_filtering = len(gest_df) - len(filtered_data_clean)
            stats_df.at[language, "Discarded during QE filtering"] = num_discarded_qe_filtering
            print(num_discarded_qe_filtering)
        else:
            print(f"File {filtered_data_path} does not exist. Skipping language {language}.")
            continue    

        # iniitalise dataframe to store unknown sentences that are not identical but don't meet heuristics 
        unknown_data = pd.DataFrame(columns=["GEST sentence", "masculine translation", "feminine translation", "original_stereotype"])
        unknown_data["GEST sentence"] = gest_df["sentence"]
        unknown_data["original_stereotype"] = gest_df["stereotype"]

        # now sort the sentences from the gendered languages into gender sensitive and gender neutral 
        for index, gest_row in filtered_data_clean.iterrows():
            masculine_translation = filtered_data.loc[index, "the man said"]
            feminine_translation = filtered_data.loc[index, "the woman said"]

            # extract quoted sentences if they've met the QE filter 
            if pd.isna(masculine_translation):
                masculine_sentence = None
            else:
                masculine_sentence = extract_quoted_sentence(masculine_translation)
            if pd.isna(feminine_translation):
                feminine_sentence = None
            else:
                feminine_sentence = extract_quoted_sentence(feminine_translation)

            ## compare two sentences and sort into neutral, gendered, and unknown
            result, words = compare_translations(masculine_sentence, feminine_sentence)
            if result == "Neutral":
                language_data.at[index, "neutral translation"] = masculine_sentence
            elif result == "Unknown":
                unknown_data.at[index, "masculine translation"] = masculine_sentence
                unknown_data.at[index, "feminine translation"] = feminine_sentence
            elif result == "Gendered":
                language_data.at[index, "masculine translation"] = masculine_sentence
                language_data.at[index, "feminine translation"] = feminine_sentence
                masc_words = []
                fem_words = []
                for gendered_word_pair in words:
                    masc_words.append(gendered_word_pair[0])
                    fem_words.append(gendered_word_pair[1])
                language_data.at[index, "gendered word masculine"] = masc_words
                language_data.at[index, "gendered word feminine"] = fem_words

            # save unknown language data to file
            os.makedirs(OUT_DIR / "discarded", exist_ok=True)
            unknown_data.to_csv(OUT_DIR / "discarded" / f"{language.lower().replace(' ', '_')}.csv")

            num_gendered_sentences = count_non_null_rows(language_data, "masculine translation")
            num_neutral_sentences = count_non_null_rows(language_data, "neutral translation")
            num_all_sentences = num_gendered_sentences + num_neutral_sentences
            num_unknown_sentences = count_non_null_rows(unknown_data, "masculine translation")
        
        num_samples_per_stereotype = count_stereotype_samples(language_data, STEREOTYPES)

        stats_df.at[language, "Neutral"] = num_neutral_sentences
        stats_df.at[language, "Gendered"] = num_gendered_sentences
        stats_df.at[language, "Discarded during heuristic filtering"] = num_unknown_sentences
        stats_df.at[language, "Coverage"] = f"{(num_all_sentences/len(language_data))*100:.2f}"
        stats_df.at[language, "Number samples per stereotype"] = str(num_samples_per_stereotype)

    # save final language data to file
    language_data.to_csv(OUT_DIR / f"{language.lower().replace(' ', '_')}_final.csv")
    
stats_df.sort_index().to_csv(OUT_DIR / f"stats_{NUM_DIFFERENT_LETTERS}_letters_{NUM_GENDERED_WORDS}_words.csv")

