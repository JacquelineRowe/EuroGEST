#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:23:54 2025

@author: s2583833
"""

import pandas as pd
from google.cloud import translate
import os
import random
import re
import string
import sys
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np

print_outputs=True

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple Metal GPU acceleration
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA GPU if available
else:
    DEVICE = torch.device("cpu") 

if print_outputs==True:
    print(f"Using device: {DEVICE}")

# load model 
model_id = "utter-project/EuroLLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

# load data
git_repo_path = '/Users/s2583833/Library/CloudStorage/OneDrive-UniversityofEdinburgh/GitRepos/genderbias'
gender_sensitive_all = pd.read_csv(f'{git_repo_path}/dataset_creation/all_gender_sensitive.csv', header=[0,1],index_col=0)
gender_insensitive_all = pd.read_csv(f'{git_repo_path}/dataset_creation/all_gender_insensitive.csv', index_col=0)
prompts_scaffolds_set = pd.read_csv(f'{git_repo_path}/model_evaluation/create_scaffolds/prompt_scaffolds_fixed_2.csv', index_col=0)

euro_llm_langs = {
                 'English': 'en',
                    'Bulgarian': 'bg',
                    'Danish': 'da',
                    'Dutch': 'nl',
                  'Estonian': 'et', 
                    'Finnish': 'fi',
                    'French': 'fr',
                     'German': 'de',
                  'Greek': 'el',
                  'Hungarian': 'hu',
                  'Irish': 'ga',
                    'Italian': 'it', 
                  'Latvian': 'lv',
                  'Lithuanian': 'lt', 
                  'Maltese': 'mt',
                   'Portuguese': 'pt',
                     'Romanian': 'ro',
                    'Spanish': 'es',
                  'Swedish': 'sv',
                  'Arabic': 'ar',
                  'Catalan': 'ca',
                #   'Chinese': 'zh',
                  'Galician': 'gl',
                #   'Hindi': 'hi',
                #   'Japanese': 'ja',
                #   'Korean': 'ko',
                  'Norwegian': 'no',
                  'Turkish': 'tr',
                   'Croatian': 'hr',
                  'Czech': 'cs',
                     'Polish': 'pl',
                  'Slovak': 'sk',
                  'Slovenian': 'sl', 
                  'Russian': 'ru',
                   'Ukrainian': 'uk'
                  }

#  function to wrap gest setnence in quotes followed by he/she or the man/woman said in that language 
def wrap_sentence(gest_sentence, language, setting):
    
    if setting == "pronouns":
        conditions = ["he said", "she said"]
    elif setting == "nouns":
        conditions = ["the man said", "the woman said"]
        
    if language == "English":
        m_scaffold = conditions[0]
        f_scaffold = conditions[1]
        
        # for non-English languages, look up the translation of he/she or the man/woman said 
        # in the prompt scaffolds table 
    else:
        m_scaffold = prompts_scaffolds_set.loc[conditions[0], language]
        f_scaffold = prompts_scaffolds_set.loc[conditions[1], language]
    
    # get rid of any punctuation at the start or end of the gest sentence (e.g. final full-stop)
    pattern = r'^[^\w\s]+|[^\w\s]+\Z'
    gest_sentence = re.sub(pattern, '', gest_sentence.strip()) 
    #  *** COMPLETE FOR OTHER LANGUAGES SPECIAL PUNCTUATION *** 
    #  maybe go back to the scaffolding set? 
    if language == "French": 
        input_text_m = f'« {gest_sentence} », {m_scaffold}. ' 
        input_text_f = f'« {gest_sentence} », {f_scaffold}. ' 
    elif language == "Lithuanian":
        input_text_m = f'„{gest_sentence}“, {m_scaffold}. ' 
        input_text_f = f'„{gest_sentence}“, {f_scaffold}. ' 
    else:
        input_text_m = f'"{gest_sentence}," {m_scaffold}. ' 
        input_text_f = f'"{gest_sentence}," {f_scaffold}. ' 
    
    return input_text_m, input_text_f


for 
