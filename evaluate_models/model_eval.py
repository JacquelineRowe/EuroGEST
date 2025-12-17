#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# redirect cache if needed for storage 
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

import re
import string
import random
import torch
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import fire
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================ #
# ================== SETUP & REPRODUCIBILITY ================= #
# ============================================================ #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple Metal GPU acceleration
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA GPU if available
else:
    DEVICE = torch.device("cpu") 

print(f"Using device: {DEVICE}")

EURO_LLM_LANGS = {
    'English': 'en', 'Bulgarian': 'bg', 'Danish': 'da', 'Dutch': 'nl',
    'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'German': 'de',
    'Greek': 'el', 'Hungarian': 'hu', 'Irish': 'ga', 'Italian': 'it',
    'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt', 'Portuguese': 'pt',
    'Romanian': 'ro', 'Spanish': 'es', 'Swedish': 'sv', 'Catalan': 'ca',
    'Galician': 'gl', 'Norwegian': 'no', 'Turkish': 'tr', 'Croatian': 'hr',
    'Czech': 'cs', 'Polish': 'pl', 'Slovak': 'sk', 'Slovenian': 'sl',
    'Russian': 'ru', 'Ukrainian': 'uk'
}

with open('punc_map.json', 'r', encoding='utf-8') as f:
    PUNC_MAP = json.load(f)

with open('prompt_scaffolds.json', 'r', encoding='utf-8') as f:
    SCAFFOLDS = json.load(f)

#============================================================== #
# ================== UTILITY FUNCTIONS========================= #
# ============================================================= #

#  function to wrap gest setnences in quotes followed he/she and the man/woman said in that language 

def wrap_sentence(sentence: str, language: str):
    """Wraps sentence in quotes and adds language-specific scaffolds."""
    m_pron, f_pron = "he said", "she said"
    m_noun, f_noun = "the man said", "the woman said"

    if language != "English":
        m_pron = SCAFFOLDS[m_pron][language]
        f_pron = SCAFFOLDS[f_pron][language]
        m_noun = SCAFFOLDS[m_noun][language]
        f_noun = SCAFFOLDS[f_noun][language]

    # Clean sentence (get rid of any punctuation at the start or end of the gest sentence (e.g. final full-stop)
    sentence = re.sub(r'^[^\w\s]+|[^\w\s]+\Z', '', sentence.strip())

    # select correct punctuation marks from dict 
    punc_start, punc_end = PUNC_MAP[language]
    
    # wrap translated gest sentence in punctuation and then create m/f noun/pronoun pairs 
    base = f"{punc_start}{sentence}{punc_end}"

    return (f"{base} {m_pron}", f"{base} {m_noun}", 
            f"{base} {f_pron}", f"{base} {f_noun}")


# new function - calculate log probs over each sequence of tokens in one forward pass
def get_sequence_log_probs(text: str, model, tokenizer):
    """
    Calculates log probabilities using a single forward pass (much faster).
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
    # Shift logits and targets so we predict the next token
    # Logits for tokens 0 to N-1 predict tokens 1 to N
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    # Calculate log_softmax over vocabulary
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # Gather the log probs of the actual tokens that appeared
    token_log_probs = torch.gather(log_probs, index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
    log_probs_list = token_log_probs[0].cpu().numpy().tolist()
    num_tokens = input_ids.shape[1]
    readable_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return log_probs_list, num_tokens, readable_tokens

# old function no longer used - more interpretable but slower as calculates log probs token by token
# def get_sequence_log_probs_old(gest_sentence: str, model, tokenizer):
#     # get rid of trailing punctuation and whitespace 
#     gest_sentence = gest_sentence.strip().rstrip(string.punctuation)
#     #  initialise list to score the log probabilities of each token 
#     log_probs_scores = []
#     tokens = tokenizer(gest_sentence, return_tensors="pt").to(DEVICE)
#     num_tokens = len(tokens['input_ids'][0])
#     for token_index in range(num_tokens-1):  # for all tokens up to last one 
#         # find which token we want to get the probabilities for from the sample sentence 
#         next_token_id = tokens['input_ids'][0][token_index+1]
#         # generate input tensor 
#         inputs = {'input_ids': tokens['input_ids'][0][:token_index+1].unsqueeze(0),
#                   'attention_mask': tokens['attention_mask'][0][:token_index+1].unsqueeze(0)}
#         readable_tokens = tokenizer.decode(inputs['input_ids'][0])
#         # predict next token 
#         outputs = model.generate(
#                         **inputs,
#                         max_new_tokens = 1,
#                         return_dict_in_generate=True,
#                         output_scores=True,
#                         # output_attentions=True,
#                         pad_token_id=model.config.eos_token_id)
#         # calculate log probabilities of all possible next tokens 
#         logits = torch.stack(outputs.scores, dim=1) 
#         log_probs_next_token = torch.nn.functional.log_softmax(logits, dim=-1)[0,0]
#         #  calculate log probability of next token of interest 
#         log_prob_next_gest_token = log_probs_next_token[next_token_id].item()
#         #  add this log probability to list of log prob scores 
#         log_probs_scores.append(log_prob_next_gest_token)
#     readable_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    
#     return log_probs_scores, num_tokens, readable_tokens

# normalise and ratio masc/fem sentence probabilities
def score_sentences(gest_sentence_m, gest_sentence_f, model, tokenizer):
    
    log_probs_scores_m, num_m_tokens, m_tokens_readable = get_sequence_log_probs(gest_sentence_m, model, tokenizer)
    sum_log_probs_m = sum(log_probs_scores_m)
    normalised_sum_log_probs_m = sum_log_probs_m - np.log(num_m_tokens)
    
    log_probs_scores_f, num_f_tokens, f_tokens_readable = get_sequence_log_probs(gest_sentence_f, model, tokenizer)    
    sum_log_probs_f = sum(log_probs_scores_f)
    normalised_sum_log_probs_f = sum_log_probs_f - np.log(num_f_tokens)

    return {
        "masc tokens": m_tokens_readable,
        "fem tokens": f_tokens_readable,
        "masc log probs": sum_log_probs_m,
        "norm masc log probs": normalised_sum_log_probs_m,
        "masc no. tokens": num_m_tokens,
        "fem log probs": sum_log_probs_f,
        "norm fem log probs": normalised_sum_log_probs_f,
        "fem no. tokens": num_f_tokens
    }


def process_results_dfs(df):
    #non normalised
    # take exp of log probs for masc and feminine
    for gender in ['masc', 'fem']:
        df[f'{gender}_probs'] = np.exp(df[f'{gender} log probs'])
    total_probs = df['masc_probs'] + df['fem_probs']
    for gender in ['masc', 'fem']:
        df[f'{gender}_prob_ratio'] = df[f'{gender}_probs'] / total_probs
        
    #  normalised 
    for gender in ['masc', 'fem']:
        df[f'norm_{gender}_probs'] = np.exp(df[f'norm {gender} log probs'])
    total_norm_probs = df['norm_masc_probs'] + df['norm_fem_probs']
    for gender in ['masc', 'fem']:
        df[f'norm_{gender}_prob_ratio'] = df[f'norm_{gender}_probs'] / total_norm_probs
    df = df.sort_values(by=["masc_prob_ratio"])
   
    return df

# ============================================================ #
# ===================== MAIN EXECUTION ======================= #
# ============================================================ #

def main(
    hf_token: str,
    hf_dataset_path: str,
    model_id: str, 
    model_label: str, 
    sample_size: int,
    languages: List[str],
    results_folder: str,
    ):

    # HF Login
    login(token=hf_token)
    print(f"Device: {DEVICE} | Model: {model_label}")

    # Load Model and data from hf 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
    model.eval()
    dataset = load_dataset(hf_dataset_path)

    for lang in languages:
        print(f"PROCESSING: {lang}")
        
        try:
            df = dataset[lang].to_pandas()
        except KeyError:
            print(f"Warning: Language {lang} not found in dataset. Skipping.")
            continue

        # Apply sampling if needed
        if sample_size < 1:
            df = df.sample(frac=sample_size)
        elif sample_size > 1:
            df = df.sample(n=min(sample_size, len(df)))

        results_list = [] 

        for idx, row in df.iterrows():
            # Standard columns in the EuroGEST HF dataset
            gest_sentence_eng = row['Source']
            gest_id = row['GEST_ID']
            stereotype_num = row['Stereotype_ID']

            # if the sentence is gendered in that language 
            if pd.notna(row.get('Masculine')) and pd.notna(row.get('Feminine')):
                m_sent = row['Masculine']
                f_sent = row['Feminine']
                condition = "G"
                # Direct scoring
                scores = score_sentences(m_sent, f_sent, model, tokenizer)
                results_list.append({
                    "ID": gest_id, "GEST sentence": gest_sentence_eng, 
                    "Stereotype no.": stereotype_num, "Condition": "G", **scores
                })

            elif pd.notna(row.get('Neutral')):
                neutral_sent = row['Neutral']

                # Wrap the neutral translation into 4 variations
                m_p, m_n, f_p, f_n = wrap_sentence(neutral_sent, lang)
                
                # Condition P: Pronoun Scaffolds
                scores_p = score_sentences(m_p, f_p, model, tokenizer)
                results_list.append({
                    "ID": gest_id, "GEST sentence": gest_sentence_eng, 
                    "Stereotype no.": stereotype_num, "Condition": "P", **scores_p
                })
                
                # Condition N: Noun Scaffolds
                scores_n = score_sentences(m_n, f_n, model, tokenizer)
                results_list.append({
                    "ID": gest_id, "GEST sentence": gest_sentence_eng, 
                    "Stereotype no.": stereotype_num, "Condition": "N", **scores_n
                })

        # Process and save results for this language
        if results_list:
            lang_results_df = pd.DataFrame(results_list)
            lang_results_df = process_results_dfs(lang_results_df)
            
            # Save to results folder
            output_path = os.path.join(results_folder, f"{lang.lower()}.csv")
            lang_results_df.to_csv(output_path, index=False)
            print(f"Saved results for {lang} to {output_path}")              
                        
if __name__ == "__main__":
    fire.Fire(main)
