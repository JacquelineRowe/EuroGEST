#!/bin/bash

# Add slurm directives and set up environment / modules as necessary if running on a cluster... 
pip install fire

# Use CuBLAS to enable deterministic behavior 
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REPO_SUBDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# change HF Home if needed
# export $HF_HOME = "/writeable/.cache/huggingface"


DEFAULT_MODEL="utter-project/EuroLLM-9B-Instruct"
DEFAULT_LABEL="EURO_LLM_9B_I"
DEFAULT_SAMPLE=3
DEFAULT_EVAL_LANG="Spanish,Russian" # the languages for evaluation. CAn also be 'all'
DEFAULT_SOURCE_LANG="English" # option to add source languages other than English
DEFAULT_PROMPTING_STRATEGY="none" # options: "none", "instruction" (task instruction only, no debiasing), 
                                # "english" (eng only debiasing), "multilingual_debiasing" (within language prompting), 
                                # "crosslingual_debiasing" (cross-lingual prompting)
DEFAULT_EVAL_TASK="translation_open" # options: "generation", "generation_MCQ", "translation", "translation_MCQ", translation_open
# e.g. you can specify from the command line which models / sample size to use
# e.g. ./model_eval.sh "meta-llama/Llama-3-8B" "LLAMA_3_8B" 10 "es"

MODEL_ID=${1:-$DEFAULT_MODEL}
MODEL_LABEL=${2:-$DEFAULT_LABEL}
SAMPLE_SIZE=${3:-$DEFAULT_SAMPLE}
EVAL_LANGUAGE=${4:-$DEFAULT_EVAL_LANG}
SOURCE_LANGUAGE=${5:-$DEFAULT_SOURCE_LANG}
PROMPTING_STRATEGY=${6:-$DEFAULT_PROMPTING_STRATEGY}
EVAL_TASK=${7:-$DEFAULT_EVAL_TASK}


HF_DATASET_PATH="utter-project/EuroGEST"
RESULTS_DIR="/writeable/model_evaluation_results/${MODEL_LABEL}"
echo $RESULTS_DIR
RESULTS_FOLDER="${RESULTS_DIR}/${EVAL_TASK}"
mkdir -p "$RESULTS_FOLDER"

# Inform the user what is being used
echo "------------------------------------------------"
echo "Running evaluation with:"
echo "  Model ID:    $MODEL_ID"
echo "  Label:       $MODEL_LABEL"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Eval Languages:    $EVAL_LANGUAGE"
echo "  Results will be saved to: ${RESULTS_FOLDER}"
echo "------------------------------------------------"

# --- 5. Language Configuration ---
if [[ "$EVAL_LANGUAGE" == "all" ]]; then
    EVAL_LANGUAGES=(
        "English" "Bulgarian" "Danish" "Dutch" "Estonian" 
        "Finnish" "French" "German" "Greek" "Hungarian" "Irish" 
        "Italian" "Latvian" "Lithuanian" "Maltese" "Portuguese" "Romanian" 
        "Spanish" "Swedish" "Catalan" "Galician" "Norwegian" "Turkish" "Croatian" 
        "Czech" "Polish" "Slovak" "Slovenian" "Russian" "Ukrainian"
    )
    echo "Evaluating for all languages."
else
    # Split the LANGUAGE string by comma into the LANGUAGES array
    IFS=',' read -ra EVAL_LANGUAGES <<< "$EVAL_LANGUAGE"
    echo "Evaluating for evaluation languages: ${EVAL_LANGUAGES[*]}"
fi

# Construct the JSON-like string for the Python argument
EVAL_LANGUAGES_STRING=$(printf '"%s", ' "${EVAL_LANGUAGES[@]}")
EVAL_LANGUAGES_STRING="[${EVAL_LANGUAGES_STRING%,}]"  

if [[ "$DEFAULT_EVAL_TASK" == "translation" || "$DEFAULT_EVAL_TASK" == "translation_MCQ" || "$DEFAULT_EVAL_TASK" == "translation_open" ]]; then
    IFS=',' read -ra SOURCE_LANGUAGES <<< "$SOURCE_LANGUAGE"
    echo "With translation from languages: ${SOURCE_LANGUAGES[*]}"
    SOURCE_LANGUAGES_STRING=$(printf '"%s", ' "${SOURCE_LANGUAGES[@]}")
    SOURCE_LANGUAGES_STRING="[${SOURCE_LANGUAGES_STRING%,}]"    
fi

python3 "$REPO_SUBDIR/model_eval_comp.py" \
--hf_token="$HUGGINGFACE_HUB_TOKEN" \
--hf_dataset_path="$HF_DATASET_PATH" \
--model_id "$MODEL_ID" \
--model_label "$MODEL_LABEL" \
--sample_size="$SAMPLE_SIZE" \
--eval_languages="$EVAL_LANGUAGES_STRING" \
--source_languages="$SOURCE_LANGUAGES_STRING" \
--results_folder="$RESULTS_FOLDER" \
--prompting_strategy="$PROMPTING_STRATEGY" \
--eval_task="$EVAL_TASK"

python3 "$REPO_SUBDIR/compare_results.py" \
--model_id "$MODEL_ID" \
--model_label "$MODEL_LABEL" \
--results_dir="$RESULTS_DIR" \
--languages="$EVAL_LANGUAGES_STRING" \


rsync -avh "$RESULTS_FOLDER/" "$HOME/results"

# Deactivate the environment when finished
deactivate
