import os
import pathlib
from typing import List

import fire
import random
import re

import git
import pandas as pd
import string
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple Metal GPU acceleration
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA GPU if available
else:
    DEVICE = torch.device("cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")

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
    'Catalan': 'ca',
    'Galician': 'gl',
    'Norwegian': 'no',
    'Turkish': 'tr',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Polish': 'pl',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Russian': 'ru',
    'Ukrainian': 'uk',
    'Arabic': 'ar',
    'Chinese': 'zh',
    'Hindi': 'hi',
    'Japanese': 'ja',
    'Korean': 'ko',
}

euro_llm_langs_punc = {
    'Bulgarian': ('„', '“,'),
    'Danish': ('"', ',"'),
    'Dutch': ('\'', ',\''),
    'English': ('"', '"'),
    'Estonian': ('"', ',"'),
    'Finnish': ('"', '",'),
    'French': ('« ', ' »,'),
    'German': ('„', '“,'),
    'Greek': ('«', '»,'),
    'Hungarian': ('„', '” –'),
    'Irish': ('\'', ',\''),
    'Italian': ('"', '",'),
    'Latvian': ('"', ',"'),
    'Lithuanian': ('“', '“,'),
    'Maltese': ('‘', ',’'),
    'Portuguese': ('"', ''),
    'Romanian': ('„', '”,'),
    'Spanish': ('“', '”,'),
    'Swedish': ('"', ',"'),
    'Catalan': ('"', '",'),
    'Galician': ('"', '"'),
    'Norwegian': ('«', ',»'),
    'Turkish': ('"', '"'),
    'Croatian': ('\'', '\','),
    'Czech': ('"', ',"'),
    'Polish': ('„', '” –'),
    'Slovak': ('„', ',“'),
    'Slovenian': ('"', ',"'),
    'Russian': ('«', '», —'),
    'Ukrainian': ('«', '», —')
    # 'Arabic': 'ar',
    # 'Chinese': 'zh',
    # 'Hindi': 'hi',
    # 'Japanese': 'ja',
    # 'Korean': 'ko',
}

result_columns = ["Language",
                  "GEST sentence",
                  "English translation",
                  "Stereotype no.",
                  "Condition",  # G = gendered, P = pronouns scaffolds,N = nouns scaffold
                  "masc tokens",
                  "fem tokens",
                  "masc log probs",
                  "norm masc log probs",
                  "masc no. tokens",
                  "fem log probs",
                  "norm fem log probs",
                  "fem no. tokens"]


#  function to wrap gest setnences in quotes followed he/she and the man/woman said in that language
def wrap_sentence(prompts_scaffolds_set, gest_sentence, language):
    m_pronoun_scaffold_eng = "he said"
    m_noun_scaffold_eng = "the man said"
    f_pronoun_scaffold_eng = "she said"
    f_noun_scaffold_eng = "the woman said"
    if language == "English":
        m_pronoun_scaffold = m_pronoun_scaffold_eng
        m_noun_scaffold = m_noun_scaffold_eng
        f_pronoun_scaffold = f_pronoun_scaffold_eng
        f_noun_scaffold = f_noun_scaffold_eng

    # for non-English languages, look up the translation of he/she or the man/woman said 
    # in the prompt scaffolds table 
    else:
        m_pronoun_scaffold = prompts_scaffolds_set.loc[m_pronoun_scaffold_eng, language]
        f_pronoun_scaffold = prompts_scaffolds_set.loc[f_pronoun_scaffold_eng, language]
        m_noun_scaffold = prompts_scaffolds_set.loc[m_noun_scaffold_eng, language]
        f_noun_scaffold = prompts_scaffolds_set.loc[f_noun_scaffold_eng, language]

    # get rid of any punctuation at the start or end of the gest sentence (e.g. final full-stop)
    pattern = r'^[^\w\s]+|[^\w\s]+\Z'
    gest_sentence = re.sub(pattern, '', gest_sentence.strip())

    # select correct punctuation marks from dict 
    start_quotes, end_quotes = euro_llm_langs_punc[language]
    # wrap translated gest sentence in punctuation and then create m/f noun/pronoun pairs 
    generic_input_text = f'{start_quotes}{gest_sentence}{end_quotes}'
    m_pronoun_input_text = f'{generic_input_text} {m_pronoun_scaffold}'
    m_noun_input_text = f'{generic_input_text} {m_noun_scaffold}'
    f_pronoun_input_text = f'{generic_input_text} {f_pronoun_scaffold}'
    f_noun_input_text = f'{generic_input_text} {f_noun_scaffold}'

    return m_pronoun_input_text, m_noun_input_text, f_pronoun_input_text, f_noun_input_text


# function to calculate conditional log probabilities of each token in a sentence
# to do -- could probably make this more efficient by passing all of the m/f identical tokens in at once at first time step? 
def generate_log_probs(tokenizer, model, gest_sentence, id_number):
    # get rid of trailing punctuation and whitespace
    gest_sentence = gest_sentence.strip().rstrip(string.punctuation)
    #  initialise list to score the log probabilities of each token 
    log_probs_scores = []

    tokens = tokenizer(gest_sentence, return_tensors="pt").to(DEVICE)
    num_tokens = len(tokens['input_ids'][0])

    batch_tokens = []
    next_token_ids = []
    for token_index in range(num_tokens - 1):  # for all tokens up to last one
        # find which token we want to get the probabilities for from the sample sentence 
        next_token_id = tokens['input_ids'][0][token_index + 1]
        # generate input tensor
        readable_tokens = tokenizer.decode(tokens['input_ids'][0][:token_index + 1], skip_special_tokens=True)
        batch_tokens.append(readable_tokens)

        next_token_ids.append(next_token_id)

    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(batch_tokens, return_tensors="pt", padding=True).to(DEVICE)

    outputs = model.generate(
        **batch,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=model.config.eos_token_id
    )

    for idx, next_token_id in enumerate(next_token_ids):
        # calculate log probabilities of all possible next tokens
        logits = outputs.scores[0][idx]
        log_probs_next_token = torch.nn.functional.log_softmax(logits, dim=-1)
        #  calculate log probability of next token of interest
        log_prob_next_gest_token = log_probs_next_token[next_token_id].item()

        #  add this log probability to list of log prob scores 
        log_probs_scores.append(log_prob_next_gest_token)

    readable_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    return log_probs_scores, num_tokens, readable_tokens


# normalise and ratio masc/fem sentence probabilities
def score_sentences(tokenizer, model, gest_sentence_m, gest_sentence_f, gest_sentence_id):
    log_probs_scores_m, num_m_tokens, m_tokens_readable = generate_log_probs(tokenizer, model, gest_sentence_m,
                                                                             f'{gest_sentence_id}_m')
    sum_log_probs_m = sum(log_probs_scores_m)
    normalised_sum_log_probs_m = sum_log_probs_m - np.log(num_m_tokens)

    log_probs_scores_f, num_f_tokens, f_tokens_readable = generate_log_probs(tokenizer, model, gest_sentence_f,
                                                                             f'{gest_sentence_id}_f')
    sum_log_probs_f = sum(log_probs_scores_f)
    normalised_sum_log_probs_f = sum_log_probs_f - np.log(num_f_tokens)

    results = [m_tokens_readable, f_tokens_readable, sum_log_probs_m, normalised_sum_log_probs_m, num_m_tokens,
               sum_log_probs_f, normalised_sum_log_probs_f, num_f_tokens]

    return results


def process_results_dfs(df):
    # non normalised
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


def main(
        model_id: str,
        model_label: str,
        results_folder: str,
        languages: List[str],
        sample_size: int = 1):
    print("MODEL ID:", model_id)
    print("MODEL LABEL:", model_label)
    print("Sample size:", sample_size)

    # load model 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

    git_repo = str(pathlib.Path(git.Repo(".", search_parent_directories=True).working_dir))
    # load data
    gender_sensitive_all = pd.read_csv(f'{git_repo}/data/gest_translations_gender_sensitive.csv',
                                       header=[0, 1],
                                       index_col=0)
    gender_insensitive_all = pd.read_csv(f'{git_repo}/data/gest_translations_gender_insensitive.csv',
                                         index_col=0)
    prompts_scaffolds_set = pd.read_csv(f'{git_repo}/data/prompt_scaffolds.csv',
                                        index_col=0)

    # make results folder if needed 
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # apply sampling if needed (1 = whole dataset) 
    if sample_size == 1:
        gender_sensitive_set = gender_sensitive_all
    else:
        gender_sensitive_set = gender_sensitive_all.sample(sample_size)

    # rename column for consistency
    gender_insensitive_set = gender_insensitive_all.rename(columns={'GEST_sentence': 'English'})

    for (language, key) in euro_llm_langs.items():
        if language not in languages:
            continue

        results_df_all = pd.DataFrame(columns=result_columns)
        print("PROCESSING: ", language)

        for index, gest_row in tqdm.tqdm(gender_sensitive_set.iterrows()):
            # extract gest sentence and sentnece id and stereotype id 
            gest_sentence = gender_sensitive_set.loc[index, "GEST_sentence"][0]
            gest_sentence_id = index
            stereotype_number = gender_sensitive_set.loc[index, "stereotype_number"].iloc[0]

            # if the sentence exists in that language in the gender sensitive set...
            if language in gender_sensitive_set.columns:
                gest_sentence_gendered_m = gender_sensitive_set.loc[index, (language, "the man said")]
                if pd.isna(gest_sentence_gendered_m):
                    pass
                # ...then calculate log probs on that sentence 
                else:
                    gest_sentence_gendered_f = gender_sensitive_set.loc[index, (language, "the woman said")]
                    results_gendered = score_sentences(tokenizer, model, gest_sentence_gendered_m,
                                                       gest_sentence_gendered_f, gest_sentence_id)
                    results_df = pd.DataFrame(
                        [[language, index, gest_sentence, stereotype_number, "G"] + results_gendered],
                        columns=results_df_all.columns)
                    results_df_all = pd.concat([results_df_all, results_df], ignore_index=True)

            # if the sentence exists in that language in the gender insensitive set...
            if language in gender_insensitive_set.columns:
                gest_sentence_translation = gender_insensitive_set.loc[index, language]
                if pd.isna(gest_sentence_translation):
                    pass
                # then wrap sentence in scaffolds and calculate log probs on scaffolded sentence 
                else:
                    gest_sentence_m_pronouns, gest_sentence_m_nouns, gest_sentence_f_pronouns, gest_sentence_f_nouns = wrap_sentence(
                        prompts_scaffolds_set, gest_sentence_translation, language)
                    results_pronouns = score_sentences(tokenizer, model, gest_sentence_m_pronouns,
                                                       gest_sentence_f_pronouns, gest_sentence_id)
                    results_nouns = score_sentences(tokenizer, model, gest_sentence_m_nouns, gest_sentence_f_nouns,
                                                    gest_sentence_id)
                    results_df_pronouns = pd.DataFrame(
                        [[language, index, gest_sentence, stereotype_number, "P"] + results_pronouns],
                        columns=results_df_all.columns)
                    results_df_nouns = pd.DataFrame(
                        [[language, index, gest_sentence, stereotype_number, "N"] + results_nouns],
                        columns=results_df_all.columns)
                    results_df_all = pd.concat([results_df_all, results_df_pronouns, results_df_nouns],
                                               ignore_index=True)

        results_df_all = process_results_dfs(results_df_all)
        # save results csv after each language is processed to backup 
        results_df_all.to_csv(f'{results_folder}/{language}.csv')


if __name__ == "__main__":
    fire.Fire(main)
