#!/bin/bash

# Add slurm directives and set up environment / modules as necessary if running on a cluster... 
pip install fire

# Use CuBLAS to enable deterministic behavior 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DEFAULT_MODEL="utter-project/EuroLLM-9B-Instruct"
DEFAULT_LABEL="EURO_LLM_9B_I"
DEFAULT_SAMPLE=3
DEFAULT_LANG="all"

REPO_SUBDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# change HF Home if needed
# export $HF_HOME = "/writeable/.cache/huggingface"


# --- Assign Arguments with Fallbacks ---
# e.g. you can specify from the command line which models / sample size to use
# e.g. ./model_eval.sh "meta-llama/Llama-3-8B" "LLAMA_3_8B" 10 "es"

MODEL_ID=${1:-$DEFAULT_MODEL}
MODEL_LABEL=${2:-$DEFAULT_LABEL}
SAMPLE_SIZE=${3:-$DEFAULT_SAMPLE}
LANGUAGE=${4:-$DEFAULT_LANG}

HF_DATASET_PATH="utter-project/EuroGEST"
RESULTS_DIR="/writeable/model_evaluation_results"
echo $RESULTS_DIR
RESULTS_FOLDER="${RESULTS_DIR}/${MODEL_LABEL}"
mkdir -p "$RESULTS_FOLDER"

# Inform the user what is being used
echo "------------------------------------------------"
echo "Running evaluation with:"
echo "  Model ID:    $MODEL_ID"
echo "  Label:       $MODEL_LABEL"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Languages:    $LANGUAGE"
echo "  Results will be saved to: ${RESULTS_FOLDER}"
echo "------------------------------------------------"

# --- 5. Language Configuration ---
if [[ "$LANGUAGE" == "all" ]]; then
    LANGUAGES=(
        "English" "Bulgarian" "Danish" "Dutch" "Estonian" 
        "Finnish" "French" "German" "Greek" "Hungarian" "Irish" 
        "Italian" "Latvian" "Lithuanian" "Maltese" "Portuguese" "Romanian" 
        "Spanish" "Swedish" "Catalan" "Galician" "Norwegian" "Turkish" "Croatian" 
        "Czech" "Polish" "Slovak" "Slovenian" "Russian" "Ukrainian"
    )
    echo "Evaluating for all languages."
else
    echo "Evaluating for language: ${LANGUAGE}"
    LANGUAGES=("${LANGUAGE}")
fi

# Construct the JSON-like string for the Python argument
LANGUAGES_STRING=$(printf '"%s", ' "${LANGUAGES[@]}")
LANGUAGES_STRING="[${LANGUAGES_STRING%,}]"  

python3 "$REPO_SUBDIR/model_eval.py" \
--hf_token="$HF_TOKEN" \
--hf_dataset_path="$HF_DATASET_PATH" \
--model_id "$MODEL_ID" \
--model_label "$MODEL_LABEL" \
--sample_size="$SAMPLE_SIZE" \
--languages="$LANGUAGES_STRING" \
--repo_subdir="$REPO_SUBDIR" \
--results_folder="$RESULTS_FOLDER"


python3 "$REPO_SUBDIR/calculate_scores.py" \
--model_id "$MODEL_ID" \
--model_label "$MODEL_LABEL" \
--results_folder="$RESULTS_FOLDER" \


rsync -avh "$RESULTS_FOLDER/" "$HOME/results"


