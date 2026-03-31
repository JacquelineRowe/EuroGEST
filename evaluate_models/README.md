# Overview
This directory contains the evaluation harness that scores a causal language model on the EuroGEST dataset, comparing the log likelihoods of gendered translations of sentences. The workflow loads a Hugging Face dataset, formats prompts per the configured scaffolds, computes log‑probabilities of masculine and feminine versions of each sentence (with option to add gender-neutral sequences too, if data is available), optionally generates continuations, and dumps per-language CSV files under results/. 

The CSV files include the sequences of log probabilities for each sentence as well as the average log probs over the tokens of interest (usually from the point of divergence until the end of the sentence, although this can be adapted depending on use case).

# Prerequisites
## 1. Python and dependencies

Use Python 3.11+.
From the repo root, run:
cd evaluate_models
pip install -r requirements.txt
This installs transformers, datasets, torch, fire, etc., with the pinned versions the harness was developed against.

## 2. Hugging Face access

Export a valid token before running the evaluation:
export HF_TOKEN="hf_..."
The script calls huggingface_hub.login() and will fail without a token.

## 3. Storage

The script creates a results folder in evaluate_models by default, but you can change this to wherever you want to save the results csv files. 

# Key files
- main.py: entry point; accepts CLI arguments via fire.Fire() and orchestrates dataset sampling, prompt generation, per-row evaluation, and result merging.
- model_eval.sh: convenience wrapper that installs fire, sets up logging, defines experiment defaults, and launches main.py with the desired settings.
- configs/: contains the language scaffolds (prompt_scaffolds.json), punctuation pairs (punc_map.json), and supported-language list (supported_languages.json) that the prompt builder uses.

# Configuration
## CLI arguments for main.py
--hf_token	Hugging Face token.
--hf_dataset_path	HF dataset ID (default utter-project/EuroGEST).
--model_id	Model identifier (Hugging Face repo) to load via AutoModelForCausalLM.
--sample_size	Number of samples per language; pass 1 to keep the whole language split.
--eval_languages	List of languages to evaluate, e.g. ["English","Spanish"].
--results_folder	Directory where per-language CSVs are written.
--seed	seed for reproducibility (42 by default).
--resume	True will merge with existing CSV if present.
--target_stereotype	none or a single stereotype ID to filter (e.g. '4')
--use_common_indices	When True, tries to sample the same GEST_IDs across languages for direct comparison.
--normalisation	Default = True (normalises sums of log probs of sequences of tokens by the number of tokens in the sequence. ). 
--measure_whole_sequences Default = False (measures only the log probs of the parts of the token sequences that differ). 

# model_eval.sh settings
DEFAULT_MODEL, DEFAULT_LABEL: choose whichever model (e.g., utter-project/EuroLLM-9B-Instruct) and label to tag the results directory.
DEFAULT_SAMPLE: 1 means “use every row” (no random sampling); any integer >1 limits the sample size per language.
DEFAULT_EVAL_LANG: comma-separated list or "all". When set to "all", the script builds a 24-language array covering the EuroGEST splits.
DEFAULT_TARGET_STEREOTYPE: none for no filtering, or something like [3,7] to limit to those stereotype IDs.

The script:

1. Sets deterministic CuBLAS behavior (CUBLAS_WORKSPACE_CONFIG).
2. Creates /writeable/job_logs and redirects stdout/stderr to a timestamped log.
3. Builds JSON lists for languages/stereotypes and prints the chosen configuration.
4. Runs python3 main.py ... with all arguments.

You can either edit the DEFAULT settings in the .sh file, or call the script with position arguments, e.g. 

```./evaluate_models/model_eval.sh "meta-llama/Llama-3-8B" "LLAMA_3_8B" 20 "English,Spanish" "2"``` 

This runs evaluation on 20 samples per language for the two languages listed.

# Running manually
If you prefer to skip the shell wrapper:

cd evaluate_models
python3 main.py \
  --hf_token="$HF_TOKEN" \
  --hf_dataset_path="utter-project/EuroGEST" \
  --model_id="model/name" \
  --sample_size=20 \
  --eval_languages='["English","German"]' \
  --results_folder="/tmp/eurogest_results" \
  --target_stereotype="2" \
  --use_common_indices=True
Remember to format eval_languages and target_stereotype as JSON-style lists or simple strings ("none").

# Output structure
- Results are saved per language as results/<MODEL_LABEL>/<EXP>/<language>.csv.
- Each row includes keys such as GEST_ID, Condition, generated log probs, and generated_text.
- The script merges new rows with existing CSVs when --resume=True, ensuring you can pick up after interruptions without duplicating data.
