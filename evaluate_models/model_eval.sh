#!/bin/bash

# Add slurm directives and set up environment / modules as necessary if running on a cluster... 
# pip install fire

# Use CuBLAS to enable deterministic behavior 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_SUBDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $REPO_SUBDIR

# specify where to save the results 
RESULTS_DIR="${REPO_SUBDIR}/results"

# change HF Home if needed
# export $HF_HOME = "/writeable/.cache/huggingface"

# you can specify from the command line the model, sample size, evaluation languages and a target stereotype (if desired) 
# e.g. ./model_eval.sh "meta-llama/Llama-3-8B" "LLAMA_3_8B" 10 "Spanish,Russian" "4" 

DEFAULT_MODEL="utter-project/EuroLLM-1.7B"
DEFAULT_LABEL="EURO_LLM_1.7B"
DEFAULT_SAMPLE=10
DEFAULT_EVAL_LANG="English,Spanish,Russian,German,Finnish" # the languages for evaluation. 'all' or a comma separated list e.g. "English,Spanish,Russian"
DEFAULT_TARGET_STEREOTYPE="none" # 'none' or numbers 1-16 for a particular stereotype

MODEL_ID=${1:-$DEFAULT_MODEL}
MODEL_LABEL=${2:-$DEFAULT_LABEL}
SAMPLE_SIZE=${3:-$DEFAULT_SAMPLE}
EVAL_LANGUAGE=${4:-$DEFAULT_EVAL_LANG}
TARGET_STEREOTYPE=${5:-$DEFAULT_TARGET_STEREOTYPE}

HF_DATASET_PATH="utter-project/EuroGEST"
RESULTS_FOLDER="${RESULTS_DIR}/${MODEL_LABEL}"
mkdir -p "$RESULTS_FOLDER"

# echo the specified settings to check
echo "------------------------------------------------"
echo "Running evaluation with:"
echo "  Model ID:           $MODEL_ID"
echo "  Sample Size:        $SAMPLE_SIZE"
echo "  Eval Languages:     $EVAL_LANGUAGE"
echo "  Target Stereotype:  $TARGET_STEREOTYPE"
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


if [[ "${TARGET_STEREOTYPE}" == "none" ]]; then
    TARGET_STEREOTYPE_STRING="none"
else
    JOINED_VALUES=$(printf ",%s" "${TARGET_STEREOTYPE[@]}")
    TARGET_STEREOTYPE_STRING="[${JOINED_VALUES:1}]"
fi

python3 "${REPO_SUBDIR}/main.py" \
--hf_token="$HF_TOKEN" \
--hf_dataset_path="$HF_DATASET_PATH" \
--model_id "$MODEL_ID" \
--sample_size="$SAMPLE_SIZE" \
--eval_languages="$EVAL_LANGUAGES_STRING" \
--results_folder="$RESULTS_FOLDER" \
--target_stereotype="$TARGET_STEREOTYPE_STRING" \

