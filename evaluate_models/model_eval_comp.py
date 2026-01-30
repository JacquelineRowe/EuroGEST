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

with open('prompting_schemas.json', 'r', encoding='utf-8') as f:
    PROMPTING_SCHEMAS = json.load(f)

def define_prompting_options(prompting_strategy, source_languages, eval_languages, eval_task):

    prompting_options = {}
    if eval_task == "translation_open":
        instruction_prompts = PROMPTING_SCHEMAS['instruction']["translation"]
    else:
        instruction_prompts = PROMPTING_SCHEMAS['instruction'][eval_task]
    debiasing_prompt_keys = PROMPTING_SCHEMAS['prompts'].keys()

    if eval_task == "generation" or eval_task == "generation_MCQ":
        languages = eval_languages
    elif eval_task == "translation" or eval_task == "translation_MCQ" or eval_task == "translation_open":
        languages = source_languages

    if prompting_strategy == "none":
        for lang in languages:
            prompting_options["baseline"] = ""
    
    if prompting_strategy == "instruction": 
        for lang in languages:
            instruction = instruction_prompts[EURO_LLM_LANGS[lang]]
            prompting_options[f"instruction-only_{EURO_LLM_LANGS[lang]}"] = instruction

    elif prompting_strategy == "english":
        instruction = instruction_prompts["en"] 
        prompting_options["instruction-only"] = instruction
        debiasing_prompts = {debiasing_prompt_key: PROMPTING_SCHEMAS['prompts'][debiasing_prompt_key]['en'] for debiasing_prompt_key in debiasing_prompt_keys}
        for debiasing_prompt_key, debiasing_prompt in debiasing_prompts.items():
            prompting_options[debiasing_prompt_key] = instruction + debiasing_prompt

        # multilinugal debiasing
    elif prompting_strategy == "multilingual" or prompting_strategy == "crosslingual":# mutilingual, within language debiasing 
        for lang in languages:
            instruction = instruction_prompts[EURO_LLM_LANGS[lang]]
            prompting_options[f"instruction-only_{EURO_LLM_LANGS[lang]}"] = instruction
            debiasing_prompts = {f"{debiasing_prompt_key}_{EURO_LLM_LANGS[lang]}": PROMPTING_SCHEMAS['prompts'][debiasing_prompt_key][EURO_LLM_LANGS[lang]] for debiasing_prompt_key in debiasing_prompt_keys}
            for debiasing_prompt_key, debiasing_prompt in debiasing_prompts.items():
                prompting_options[debiasing_prompt_key] = instruction + debiasing_prompt
        
    return prompting_options


#============================================================== #
# ================== UTILITY FUNCTIONS========================= #
# ============================================================= #

#  function to wrap gest setnences in quotes followed he/she and the man/woman said in that language 

def wrap_neutral_sentence(sentence: str, language: str):
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


# normalise and ratio masc/fem sentence probabilities
def score_sentences_constrained(gest_sentence_m, gest_sentence_f, model, tokenizer):
    
    log_probs_scores_m, num_m_tokens, m_tokens_readable = get_sequence_log_probs(gest_sentence_m, model, tokenizer)
    sum_log_probs_m = sum(log_probs_scores_m)
    normalised_sum_log_probs_m = sum_log_probs_m - np.log(num_m_tokens)
    
    log_probs_scores_f, num_f_tokens, f_tokens_readable = get_sequence_log_probs(gest_sentence_f, model, tokenizer)    
    sum_log_probs_f = sum(log_probs_scores_f)
    normalised_sum_log_probs_f = sum_log_probs_f - np.log(num_f_tokens)

    # convert to probabilities 
    masc_probs = np.exp(normalised_sum_log_probs_m)
    fem_probs = np.exp(normalised_sum_log_probs_f)

    total_probs = masc_probs + fem_probs
    masc_prob_ratio = masc_probs / total_probs

    return {
        "masc_tokens": m_tokens_readable,
        "fem_tokens": f_tokens_readable,
        # "masc log probs": sum_log_probs_m,
        f"masc_prob_ratio": masc_prob_ratio,
        # "masc no. tokens": num_m_tokens,
        # "fem log probs": sum_log_probs_f,
        # f"fem_probs": fem_probs,
        # "fem no. tokens": num_f_tokens
    }

def score_sentences_MCQ(sent_combined_prompt, model, tokenizer):
    
    inputs = tokenizer(sent_combined_prompt, return_tensors="pt").to(DEVICE)
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=1, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7 # add this as an experiment variable
    )

    # 2. Extract and decode the single new token
    input_length = inputs.input_ids.shape[1]
    raw_token = tokenizer.decode(output_tokens[0][input_length:])
    
    return raw_token



def generate_new_tokens(text, model, tokenizer, num_tokens):
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=num_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7 # add this as an experiment variable
    )

    # 2. Extract and decode the single new token
    input_length = inputs.input_ids.shape[1]
    raw_tokens = tokenizer.decode(output_tokens[0][input_length:])
    
    return raw_tokens
    
    


def get_consistent_indices(dataset, eval_languages, sample_size, seed=42):
    """
    Finds common indices across eval_languages. 
    Falls back to independent sampling if no intersection exists.
    """
    random.seed(seed)
    common_indices = None
    
    # 1. Attempt to find the intersection
    for lang in eval_languages:
        try:
            current_indices = set(dataset[lang].to_pandas()["GEST_ID"])
            if common_indices is None:
                common_indices = current_indices
            else:
                common_indices = common_indices.intersection(current_indices)
        except KeyError:
            continue

    # 2. Logic for Shared Indices
    if common_indices and len(common_indices) > 0:
        # SORT the list to make the conversion from set deterministic
        common_list = sorted(list(common_indices))

        if sample_size == 1:
            n = len(common_list)
        if sample_size > 1:
            n = min(int(sample_size), len(common_list))
        
        sampled = random.sample(common_list, n)
        print(f"SUCCESS: Found {len(common_indices)} shared rows. Sampled {len(sampled)}.")
        # Return a dictionary where every language points to the same list
        return {lang: sampled for lang in eval_languages}

    # 3. Fallback: Independent Random Sampling
    print("WARNING: No common indices found. Reduce sample size or number of eval_languages.")
    fallback_map = {}
    for lang in eval_languages:
        df_len = len(dataset[lang])
        all_ids = list(range(df_len))
        
        if sample_size == 1:
            n = sample_size
        else:
            n = min(int(sample_size), df_len)
            
        fallback_map[lang] = random.sample(all_ids, n)
    
    return fallback_map

def evaluate_sentence(row, gendered_row, prompting_options, model, tokenizer, lang, eval_task):
    ## TO DO: sort out prompting instructions so that we only add the language specifier when there are instructions present 
    results_dict = {}
    eng_sentence = row['Source'] # N.B. when adding multiple source langauges will need to reconfigure 
    if eval_task == "translation_MCQ":
        extra_prompt = f"\"{eng_sentence} {EURO_LLM_LANGS[lang]}: "
    elif eval_task == "translation" or eval_task == "translation_open":
        extra_prompt = f"{lang}: \"{eng_sentence} {EURO_LLM_LANGS[lang]}: "
    else:
        extra_prompt = ""

    if gendered_row == True:
        m_sent = row['Masculine']
        f_sent = row['Feminine']
        condition = "G"

    elif gendered_row == False:
        neutral_sent = row['Neutral']
        # Wrap the neutral translation into 4 variations
        m_sent, m_sent_noun, f_sent, f_sent_noun = wrap_neutral_sentence(neutral_sent, lang)
        condition = "P"
        # if no gendered pronouns, use noun scaffolds instead 
        if m_sent == f_sent:
            m_sent = m_sent_noun
            f_sent = f_sent_noun 
            condition = "N"

    if eval_task == "generation" or eval_task == "translation":
        # if generating, we can simply add whatever instructions or debiasing prompts are required to the start of the masc and fem sentence 
        for prompt_id, prompt_strategy in prompting_options.items():
            m_sent_prompt = prompt_strategy + extra_prompt + m_sent
            f_sent_prompt = prompt_strategy + extra_prompt + f_sent

            scores = score_sentences_constrained(m_sent_prompt, f_sent_prompt, model, tokenizer) 
            results_dict[f"masc_prob_ratio_{prompt_id}"] = scores["masc_prob_ratio"]
    
    elif eval_task == "translation_open":

        # find gendered term in gendered sentence pair 
        masc_words = m_sent.split()
        for idx, word in enumerate(f_sent.split()):
            # to do - debug (e.g. for spanish)
            if word not in masc_words:
                feminine_word = word
                masculine_word = masc_words[idx]
                break
        try:
            feminine_word
        except NameError:
            print("no feminine word found", m_sent, f_sent)
        else:
            for prompt_id, prompt_strategy in prompting_options.items():
                translation_prompt = prompt_strategy + extra_prompt
                # generate new tokens 
                translation_hypothesis = generate_new_tokens(translation_prompt, model, tokenizer, 50)
                if feminine_word in translation_hypothesis and masculine_word not in translation_hypothesis:
                    results_dict[f"masc_gender_select_{prompt_id}"] = 0
                elif masculine_word in translation_hypothesis and feminine_word not in translation_hypothesis:
                    results_dict[f"masc_gender_select_{prompt_id}"] = 1
                else:
                    results_dict[f"masc_gender_select_{prompt_id}"] = "Error"
            
    elif eval_task == "generation_MCQ" or eval_task == "translation_MCQ":
        sentences = {"M": m_sent, "F": f_sent}
        ordering = ["M", "F"]
        # combine m_sent and f_sent into a single prompt with MCQ options, randomising the order of masc / fem in options 1 and 2 
        # QUESTION: Is it enoguh to randomise, or should I do both cases flipped? 
        random.shuffle(ordering)

        option_1 = ordering[0]
        option_2 = ordering[1]

        sent_combined = f"1. {sentences[option_1]} 2. {sentences[option_2]}"
        for prompt_id, prompt_strategy in prompting_options.items():
            sent_combined_prompt = prompt_strategy + extra_prompt + sent_combined + ". Answer: "
            selected_option = generate_new_tokens(sent_combined_prompt, model, tokenizer, 1)

            selected_gender = None

            if selected_option == "1":
                selected_gender = option_1
            elif selected_option == "2":
                selected_gender = option_2
            else:
                results_dict[f"masc_selection_{prompt_id}"] = "Error"

            if selected_gender:
                if selected_gender == "M":
                    results_dict[f"masc_selection_{prompt_id}"] = 1
                elif selected_gender == "F":
                    results_dict[f"masc_selection_{prompt_id}"] = 0
                else:
                    results_dict[f"masc_selection_{prompt_id}"] = "Error"

        # scores = score_sentences_MCQ(sent_combined_prompt, model, tokenizer)

        # results_dict[f"masc_prob_ratio_{prompt_id}"] = scores["masc_prob_ratio"]

    return results_dict, condition

# ============================================================ #
# ===================== MAIN EXECUTION ======================= #
# ============================================================ #

def main(
    hf_token: str,
    hf_dataset_path: str,
    model_id: str, 
    model_label: str, 
    sample_size: int,
    eval_languages: List[str],
    source_languages: List[str],
    results_folder: str,
    prompting_strategy: bool,
    eval_task: str = "generation"
    ):

    # HF Login
    login(token=hf_token)
    print(f"Device: {DEVICE} | Model: {model_label}")

    output_dir =  f'{results_folder}'
    os.makedirs(output_dir, exist_ok=True)

    # Load Model and data from hf 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
    model.eval()
    dataset = load_dataset(hf_dataset_path)
    
    # get common indices 
    sampled_indices = get_consistent_indices(dataset, eval_languages, sample_size, seed=SEED)
    prompting_options_all = define_prompting_options(prompting_strategy, source_languages, eval_languages, eval_task)

    for eval_lang in eval_languages:
        print(f"PROCESSING: {eval_lang}")

        if prompting_strategy == "multilingual":
            prompting_options = {prompt_id: prompting_options_all[prompt_id] for prompt_id in prompting_options_all.keys() if prompt_id.endswith(EURO_LLM_LANGS[eval_lang])}
        else:
            prompting_options = prompting_options_all

        # regardless of condition, include baseline with no prompt too 
        # prompting_options["baseline"] = "" 

        output_path = os.path.join(output_dir, f"{eval_lang.lower()}.csv")

        if os.path.exists(output_path):
            print(f"Loading existing results for {eval_lang} at {output_path}. WARNING: may overwrite existing results.")
            existing_results = pd.read_csv(output_path)
        else:
            existing_results = []

        try:
            lang_df = dataset[eval_lang].to_pandas()
        except KeyError:
            print(f"Warning: Language {eval_lang} not found in dataset. Skipping.")
            continue

        # Apply sampling if needed
        # print("Sampled indices for language ", lang, sampled_indices[lang])
        df = lang_df[lang_df['GEST_ID'].isin(sampled_indices[eval_lang])]

        results = {gest_id: {"GEST_ID": gest_id} for gest_id in df['GEST_ID']}

        for idx, row in df.iterrows():
            # Standard columns in the EuroGEST HF dataset
            gest_sentence_eng = row['Source']
            gest_id = row['GEST_ID']
            stereotype_num = row['Stereotype_ID']

            # if the sentence is gendered in that language 
            if pd.notna(row.get('Masculine')) and pd.notna(row.get('Feminine')):
                gendered_row = True
            elif pd.notna(row.get('Neutral')):
                gendered_row = False 
                # if we are doing translation and the target language has no gendered forms, we skip this example
                if eval_task == "translation" or eval_task == "translation_MCQ" or eval_task == "translation_open":
                    results[gest_id] = {
                        "GEST_ID": gest_id, 
                        "GEST sentence": gest_sentence_eng, 
                        "Stereotype no.": stereotype_num}
                    continue
            
            results_dict, condition = evaluate_sentence(row, gendered_row, prompting_options, model, tokenizer, eval_lang, eval_task)
            results[gest_id] = {
                "GEST_ID": gest_id, 
                "GEST sentence": gest_sentence_eng, 
                "Stereotype no.": stereotype_num, 
                "Condition": condition, 
                # This adds all the prompt-specific results (e.g., masc_gender_select_instruction)
                **results_dict 
            }
        
        # Process and save results for this language
        if results:
            lang_results_df = pd.DataFrame(list(results.values()))
            # lang_results_df = lang_results_df.sort_values(by=["masc_prob_ratio_baseline"])

            if len(existing_results) > 1:
                # 1. Identify the columns that are ONLY in the new dataframe
                new_cols = [c for c in lang_results_df.columns if c not in existing_results.columns]
                print(new_cols)
                assert len(existing_results) == len(lang_results_df)
                assert(list(existing_results["GEST_ID"]) == list(lang_results_df["GEST_ID"]))
                # 2. Add those columns to the existing dataframe using the index
                # This ignores matching IDs and just glues them together row-by-row
                for col in new_cols:
                    existing_results[col] = lang_results_df[col]

            else:
                existing_results = lang_results_df

            existing_results.to_csv(output_path, index=False)
            print(f"Saved results for {eval_lang} to {output_path}")    
            print(existing_results.head())          
                        
if __name__ == "__main__":
    fire.Fire(main)
