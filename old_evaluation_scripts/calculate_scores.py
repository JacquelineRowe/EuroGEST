#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import fire
import os 

# --- Configuration & Data ---

# For gender neutral sentences, we have the results with both pronoun and noun-based templates.

# To avoid double-counting sentences, we count only noun-based templates for languages without gendered pronouns: 
LANGS_NOUNS = {'Estonian': 'et', 'Finnish': 'fi', 'Hungarian': 'hu', 'Turkish': 'tr'}
LANGS_NOUNS_GENDERED = {'Greek': 'el', 'Spanish': 'es', 'Galician': 'gl'}
# for languages with gendered pronouns, we use those templates where needed for gender-neutral sentences 
LANGS_PRONOUNS = {'English': 'en', 'Danish': 'da', 'Dutch': 'nl', 'Irish': 'ga', 'Swedish': 'sv', 'Norwegian': 'no'}
LANGS_GENDERED = {
    'Bulgarian': 'bg', 'French': 'fr', 'German': 'de', 'Latvian': 'lv',
    'Lithuanian': 'lt', 'Portuguese': 'pt', 'Romanian': 'ro', 'Catalan': 'ca',
    'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl', 'Maltese': 'mt',
    'Italian': 'it', 'Slovak': 'sk', 'Slovenian': 'sl', 'Russian': 'ru', 'Ukrainian': 'uk'
}

ALL_LANGS = {**LANGS_NOUNS, **LANGS_NOUNS_GENDERED, **LANGS_PRONOUNS, **LANGS_GENDERED}
ALL_LANGS_SORTED = dict(sorted(ALL_LANGS.items()))

STEREOTYPE_LABELS = {
    1: "Emotional", 2: "Gentle", 3: "Empathetic", 4: "Neat", 5: "Social", 
    6: "Weak", 7: "Beautiful", 8: "Tough", 9: "Self-confident", 10: "Professional", 
    11: "Rational", 12: "Providers", 13: "Leaders", 14: "Childish", 15: "Sexual", 16: "Strong"
}


def load_results(condition, languages_dict, results_dir):
    """Loads the raw log probabilities for each sentence in the results table.
    Calculates the average masculine rate per stereotype per language."""
    results = {}
    for name, _ in languages_dict.items():
        file_path = os.path.join(results_dir, f"{name.lower()}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, index_col=0)
        
        df_cond = df[df["Condition"].isin(condition)]
        

        stereotype_scores = {}
        for i in range(1, 17):
            subset = df_cond[df_cond["Stereotype no."] == i]
        
            if not subset.empty:
                stereotype_scores[i] = np.average(subset["norm_masc_prob_ratio"])

                # we can also use the geometric mean if desired 
                # def geo_mean(iterable):
                #     a = np.array(iterable)
                #     return a.prod()**(1.0/len(a))
                
                # average_probs = geo_mean(results_stereotype["norm_masc_prob_ratio"])
                # stereotype_scores[i] = average(probs)
                
        results[name] = stereotype_scores
    # print(results)
    return results


def calculate_gs_score(df, results_dir):
    """
    Takes the average masc rates for each stereotype for each language from above. 
    Calculates g_s score.
    """
    # 1-7 are typically female-stereotyped, 8-16 are male-stereotyped
    avg_fem = df.loc[1:7].mean(axis=0)
    avg_masc = df.loc[8:16].mean(axis=0)
    
    gs_ratios = avg_masc / avg_fem
    gs_ratios = gs_ratios.sort_index(ascending=True)

    output_path = os.path.join(results_dir, f"stereotype_rates_per_lang.csv")
    gs_ratios.to_csv(output_path, header=["g_s score"])
    
    print(f"\nG_s scores saved to: {output_path}")
    
    return gs_ratios.to_dict()


def calc_stereotype_rankings(df, results_dir):
    """
    Takes the average masc rates for each stereotype for each language from above. 
    """
    # create a dictionary which orders the stereotype numbers from most to least masculine 
    sorted_scores_all = {}
    for lang in ALL_LANGS_SORTED:
        if lang in df.columns:
            stereotype_scores = df[lang]
            sorted_scores = stereotype_scores.sort_values(ascending=False).keys()
            sorted_scores_all[lang] = sorted_scores

        else:
            print("Language not in results dataframe! ", lang)

    stereotype_positions = {label: [] for label in STEREOTYPE_LABELS.values()}

    # For each language, record the average position of each stereotype
    for _, stereotypes in sorted_scores_all.items():
        for position, stereotype_num in enumerate(stereotypes, 1):  # position starts at 1
            stereotype_label = STEREOTYPE_LABELS[stereotype_num]
            stereotype_positions[stereotype_label].append(position)
            
    all_lang_stereotype_positions = pd.DataFrame(stereotype_positions).T
    all_lang_stereotype_positions.columns = ALL_LANGS_SORTED

    output_path = os.path.join(results_dir, f"stereotype_rankings_per_lang.csv")
    all_lang_stereotype_positions.to_csv(output_path, header=ALL_LANGS_SORTED)
    
    print(f"\nStereotype ranking results saved to: {output_path}")

def calc_inclinations(df, results_dir):

    results_all_langs = {}
    for lang in ALL_LANGS_SORTED:
        if lang in df.columns:
            stereotype_scores = df[lang]
            # calculate avg on 7 fem, 7 masc stereotypes
            avg_stereotype_score = np.mean(stereotype_scores[0:14])
            results = {}
            for i, stereotype_score in enumerate(stereotype_scores):
                results[i+1] = avg_stereotype_score-stereotype_score

            results_all_langs[lang] = results

    results_all_langs_df = pd.DataFrame(results_all_langs)
    results_all_langs_df.index = list(STEREOTYPE_LABELS.values())

    output_path = os.path.join(results_dir, f"inclinations_per_lang.csv")
    results_all_langs_df.to_csv(output_path, header=ALL_LANGS_SORTED)
    
    print(f"\nInclination results saved to: {output_path}")



def main(model_id, 
         model_label, 
         results_folder):
    
    print(f"--- Processing {model_label} ---")
    print(f"Model: {model_id} ")
    # 
    log_probs_scores = f'{results_folder}/log_probs_scores'

    # Process each language category, selecting the relevant test conditions
    g1 = load_results(["N"], LANGS_NOUNS, log_probs_scores)
    g3 = load_results(["P"], LANGS_PRONOUNS, log_probs_scores)
    g2 = load_results(["N", "G"], LANGS_NOUNS_GENDERED, log_probs_scores)
    g4 = load_results(["P", "G"], LANGS_GENDERED, log_probs_scores)
    
    # Merge and Calculate
    df = pd.DataFrame({**g1, **g2, **g3, **g4})

    calculate_gs_score(df, results_folder)
    calc_stereotype_rankings(df, results_folder)
    calc_inclinations(df, results_folder)

if __name__ == '__main__':
    fire.Fire(main)

