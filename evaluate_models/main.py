import os
import fire
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

pd.reset_option('display.max_colwidth')

# Import utils
from utils import (
    setup_environment, 
    load_scaffolds_configs, 
    SUPPORTED_LANGS, 
    get_consistent_indices, 
    format_target_stereotype,
    define_prompting_options, 
    build_row_prompts,
    get_sequence_log_probs,
    find_num_diff_idx,
    tokenise,
    generate_new_tokens,
)

def evaluate_sentence(model_inputs, model, tokenizer, device, normalisation):

    model_inputs_m = model_inputs[0]
    model_inputs_f = model_inputs[1]

    inputs_m, n_m, t_m = tokenise(model_inputs_m, tokenizer, device)
    inputs_f, n_f, t_f = tokenise(model_inputs_f, tokenizer, device)
    
    # for full sentence translations, multiple tokens may differ; for a MCQ question, only the final answer tokens will differ 
    diff_indices = [i for i in range(min(len(t_m), len(t_f))) if t_m[i] != t_f[i]]
    # check also for cases where one sentence is actually contained in the other longr one 
    if not diff_indices and len(t_m) != len(t_f):               
        start = max(len(t_m), len(t_f)) - (max(len(t_m), len(t_f)) - min(len(t_m), len(t_f))) - 2 # we have to back up 2 tokens so that we calculate probs on last portion of shorter sentence as well as longer one 
    else:
        # Standard logic: back up one token from the first difference for context
        start = diff_indices[0] - 1 if diff_indices else 0 # in teh standard case we start evals from point of divergence -1 

    # get log probs of divergent portions of sequences
    lp_m = get_sequence_log_probs(inputs_m["input_ids"], model, device, start_index=start)
    lp_f = get_sequence_log_probs(inputs_f["input_ids"], model, device, start_index=start)

    if len(model_inputs) > 2:
        model_inputs_n = model_inputs[2]
        input_ids_n, n_n, t_n = tokenise(model_inputs_n, tokenizer, device)
        lp_n = get_sequence_log_probs(input_ids_n["input_ids"], model, device, start_index=start)
    else:
        lp_n, n_n, t_n = None, None, None

    # sum over the log probs 
    lp_m_sum = sum(lp_m) if isinstance(lp_m, (list, np.ndarray)) else lp_m
    lp_f_sum = sum(lp_f) if isinstance(lp_f, (list, np.ndarray)) else lp_f 
    lp_n_sum = sum(lp_n) if isinstance(lp_n, (list, np.ndarray)) else lp_n

    if n_m != n_f:
        num_diff_m, num_diff_f = find_num_diff_idx(t_m, t_f, start)
    else:
        num_diff_m, num_diff_f = (1,1)

    # Normalized by length of the DIFFERING part only
    lp_m_avg = (lp_m_sum / num_diff_m) if normalisation else lp_m_sum
    lp_f_avg = (lp_f_sum / num_diff_f) if normalisation else lp_f_sum
    lp_n_avg = None if normalisation else lp_n_sum # TO DO: deal with gender-neutral translations in normalisation 

    # convert log probs into probs
    prob_m = np.exp(lp_m_avg)
    prob_f = np.exp(lp_f_avg)
    prob_n = np.exp(lp_n_avg) if lp_n_avg is not None else None

    ## Also do extrinsic evaluation by generating novel text
    # we only want to generate roughly as many tokens as we actually need to answer the question
    num_different_tokens = max(len(t_m), len(t_f)) - start 
    tokens_to_generate = max(num_different_tokens+3, 5) # we set a minimum of 5 tokens to generate to ensure we get a meaningful continuation, even if only one token differs in the original sentences

    identical_tokens = {k: v[:, :start] for k, v in inputs_m.items()}
    # using greedy decoding for now
    generated_text = generate_new_tokens(identical_tokens, model, tokenizer, tokens_to_generate, device, do_sample=False)

    return {
        "masc_tokens": t_m, 
        "fem_tokens": t_f, 
        "neutral_tokens": t_n,
        "masc_log_probs_sequence": lp_m,
        "fem_log_probs_sequence": lp_f,
        "neutral_prob_sequence": lp_n,
        "masc_prob": prob_m,
        "fem_prob": prob_f,
        "neutral_prob": prob_n,
        "identical_tokens": t_m[:start],
        "generated_text": generated_text
    }


def main(hf_token, 
         hf_dataset_path, 
         model_id, 
         sample_size, 
         eval_languages, 
         source_languages, 
         results_folder, 
         exp="generation", 
         seed=42,
         resume=False,
         target_stereotype="none",
         use_common_indices=False, # e.g. if you want to only select the common seentences from the languages of evaluation for direct comparisons
         normalisation=True): # set as 1 for task/debiasing experiments, set as 2 for translation-specific spectrum experiments 
    
   
    # 1. Setup
    device = setup_environment(seed)
    login(token=hf_token)
    os.makedirs(results_folder, exist_ok=True)

    # 2. Load Configs using our new helper
    punc_map, scaffolds = load_scaffolds_configs()

    # 3. Initialize Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Optimization: ensure padding is correct for batching if needed later
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    dataset = load_dataset(hf_dataset_path)

    # 4. Prepare Indices and Prompts
    if exp == "translation":
        try:
            assert source_languages is not None, "Error: 'source_languages' must be specified for translation experiments."
            if isinstance(source_languages, str) and source_languages != '':
                language_set = eval_languages + [source_languages]
            elif isinstance(source_languages, list):
                language_set = eval_languages + source_languages 
        except AssertionError as e:            print(e)
    else:
        language_set = eval_languages

    if use_common_indices:
        sampled_indices = get_consistent_indices(dataset, language_set, sample_size, target_stereotype, seed=seed)
    else:
        sampled_indices = {}
        for lang in language_set:
            df_len = len(dataset[lang])
            all_ids = list(range(df_len))
            n = df_len if sample_size == 1 else min(int(sample_size), df_len)
            sampled_indices[lang] = random.sample(all_ids, n)

    # use prompting function to define set of variables prompts 
    prompting_options = define_prompting_options(source_languages, eval_languages, exp)

    # 5. Process Languages
    for eval_lang in eval_languages:
        print(f"PROCESSING: {eval_lang}")
        # --- A. Handle Dataset Loading First ---
        try:
            # We must define 'df' here so it is specific to the current eval_lang
            df = dataset[eval_lang].to_pandas()
            df = df[df['GEST_ID'].isin(sampled_indices[eval_lang])]
           
            if df.empty:
                print(f"No rows found for {eval_lang} with the specified criteria. Skipping.")
                continue

        except KeyError:
            print(f"Skipping {eval_lang}: not in dataset.")
            continue

        output_path = os.path.join(results_folder, f"{eval_lang.lower()}.csv")
        
        # 5. Load existing ONLY if resume flag is set
        if resume and os.path.exists(output_path):
            existing_results = pd.read_csv(output_path)
            print(f"Resuming: Loaded {len(existing_results)} existing rows for {eval_lang}")
        else:
            existing_results = pd.DataFrame()
            if os.path.exists(output_path) and not resume:
                print(f"Note: {output_path} exists but 'resume' is False. Overwriting.")

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {eval_lang}"):

            if device.type == 'mps':
                torch.mps.empty_cache()
                
            is_gendered = pd.notna(row.get('Masculine')) and pd.notna(row.get('Feminine'))
            is_neutral = pd.notna(row.get('Neutral'))
            
            if not is_gendered and not is_neutral:
                continue
            
            # Skip translation for purely neutral if no gendered forms exist
            if not is_gendered and exp == "translation":
                continue   
                
            prompting_inputs, cond = build_row_prompts(row, is_gendered, prompting_options, eval_lang, scaffolds, punc_map, exp)

            if prompting_inputs is None:
                print(f"Skipping {row}")
                continue

            for prompt_id, model_inputs in prompting_inputs.items():
                scores = evaluate_sentence(model_inputs, model, tokenizer, device, normalisation)
                results.append({
                    "GEST_ID": row['GEST_ID'],
                    "Source": row['Source'],
                    "Stereotype_ID": row['Stereotype_ID'],
                    "Condition": cond,
                    "Prompt ID": prompt_id,
                    **scores,
                })

        
        # 6. Save Results
        if results:
            new_df = pd.DataFrame(results)
            join_keys = ["GEST_ID", "Source", "Stereotype_ID", "Condition"]

            if not existing_results.empty:
                # Remove columns from existing_results that exist in new_df 
                # (except for the join keys) to avoid _x / _y suffixes.
                cols_to_drop = [c for c in new_df.columns if c in existing_results.columns and c not in join_keys]
                existing_results = existing_results.drop(columns=cols_to_drop)
                
                final_df = existing_results.merge(new_df, on=join_keys, how="outer")
            else:
                final_df = new_df
            
            # pd.set_option('display.max_colwidth', 25)
            print(final_df)

            final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    fire.Fire(main)
