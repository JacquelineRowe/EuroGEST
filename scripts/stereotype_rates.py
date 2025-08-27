#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:06:26 2025

@author: s2583833
"""


import pandas as pd
import os
import random
import re
import string
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# folder where the results csvs are stored 
results_dir = "/net/storage/pr3/plgrid/plggmultilingualnlp/mklimasz/EuroGEST/scripts/results"

#  languages with only gendered nouns
langs_nouns = {
    'Estonian': 'et', 
    'Finnish': 'fi',
    'Hungarian': 'hu',
    }

#  langauges with only gendered nouns and gendered words 
langs_nouns_gendered = {
    'Greek': 'el',
    'Spanish': 'es',
    'Italian': 'it'
    }

# languages with gendered pronouns but no gendered words  
langs_pronouns = {
    'English': 'en',
    'Danish': 'da',
    'Dutch': 'nl',
    'Irish': 'ga',
    'Swedish': 'sv',
    'Norwegian': 'no',
    }

# languages with gendered pronouns and gendered words 
langs_gendered = {
    'Bulgarian': 'bg',
    'French': 'fr',
    'German': 'de',
    'Latvian': 'lv',
    'Lithuanian': 'lt', 
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Catalan': 'ca',
    'Galician': 'gl',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Polish': 'pl',
    'Slovak': 'sk',
    'Slovenian': 'sl', 
    'Russian': 'ru',
    'Ukrainian': 'uk'
 }

all_langs = langs_nouns | langs_nouns_gendered | langs_pronouns | langs_gendered
keys = list(all_langs())

    
def load_results(condition,languages):
    results = dict.fromkeys(languages)
    # extract results for that language and that condition
    for language in languages:
        results_lang_only = pd.read_csv(f'{results_dir}/{language}.csv', index_col=0)
        results_lang_condition_only = results_lang_only[results_lang_only["Condition"].isin(condition)]
           
        stereotype_scores = {}
        for stereotype_number in range(1,17):
            
            #  extract results for each stereotype number in the masc/fem set 
            results_stereotype = results_lang_condition_only.loc[results_lang_condition_only["Stereotype no."] == stereotype_number]
          
            if not results_stereotype.empty:
                try:
                    # compute the average of the specific column (results_key)
                    average_probs = np.average(results_stereotype["norm_masc_prob_ratio"])
                    stereotype_scores[stereotype_number] = average_probs
                    # Store the result
                except RuntimeWarning as e:
                    # Print a message if there's a runtime warning
                    print(f"Warning while calculating average for stereotype {stereotype_number} and language {language}: {e}")

            else:
                # If results_stereotype is empty, print a message
                print(f"No {condition} data for stereotype {stereotype_number} and language {language}")

        results[language] = stereotype_scores
            
    return results


def calc_results(results_dir):
    
    model_results = {}
        
    group_1 = load_results(["N"],langs_nouns)
    group_2 = load_results(["N", "G"], langs_nouns_gendered)
    group_3 = load_results(["P"], langs_pronouns)
    group_4 = load_results(["P", "G"], langs_gendered)
    
    all_langs_df = pd.DataFrame(group_1 | group_2 | group_3 | group_4)
    
    # load results for each stereotype
    avg_fem = all_langs.loc[1:7].mean(axis=0)
    # std_fem = all_langs.loc[1:7].std(axis=0)
    avg_masc = all_langs.loc[8:16].mean(axis=0)
    # std_masc = all_langs.loc[8:16].std(axis=0)
    
    gs_ratios = []
    languages = list(avg_fem.keys())
    for index in range(0, len(languages)):
        gs_ratio = avg_masc[index] / avg_fem[index]
        gs_ratios.append(gs_ratio)
    
    return gs_ratios, all_langs


model_results, all_langs = calc_results(results_dir)

