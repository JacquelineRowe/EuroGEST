#!/bin/bash

# Add slurm directives and set up environment / modules as necessary if running on a cluster... 
pip install fire

# Use CuBLAS to enable deterministic behavior 
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REPO_SUBDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $REPO_SUBDIR

# change HF Home if needed
# export $HF_HOME = "/writeable/.cache/huggingface"

DEFAULT_MODEL="utter-project/EuroLLM-1.7B-Instruct"
DEFAULT_LABEL="EURO_LLM_1.7B_I"
DEFAULT_SAMPLE=10
DEFAULT_EVAL_LANG="Russian,Spanish,German" # the languages for evaluation. 'all' or a comma separated list e.g. "English,Spanish,Russian"
DEFAULT_SOURCE_LANG="English" # option to add source languages other than English
DEFAULT_EXP="debiasing_multilingual" # options: "none" (regular evalution), "translation" "debiasing_english", "debiasing_multilingual"
DEFAULT_TARGET_STEREOTYPE=4 # 'none' or numbers 1-16 for a particular stereotype
# e.g. you can specify from the command line which models / sample size to use
# e.g. ./model_eval.sh "meta-llama/Llama-3-8B" "LLAMA_3_8B" 10 "es"

MODEL_ID=${1:-$DEFAULT_MODEL}
MODEL_LABEL=${2:-$DEFAULT_LABEL}
SAMPLE_SIZE=${3:-$DEFAULT_SAMPLE}
EVAL_LANGUAGE=${4:-$DEFAULT_EVAL_LANG}
SOURCE_LANGUAGE=${5:-$DEFAULT_SOURCE_LANG}
EXP=${6:-$DEFAULT_EXP}
TARGET_STEREOTYPE=${7:-$DEFAULT_TARGET_STEREOTYPE}


HF_DATASET_PATH="utter-project/EuroGEST"
RESULTS_DIR="../model_evaluation_results_modular/${MODEL_LABEL}"
echo $RESULTS_DIR
RESULTS_FOLDER="${RESULTS_DIR}/${EXP}"
mkdir -p "$RESULTS_FOLDER"

# Inform the user what is being used
echo "------------------------------------------------"
echo "Running aluation with:"
echo "  Model ID:           $MODEL_ID"
echo "  Sample Size:        $SAMPLE_SIZE"
echo "  Eval Languages:     $EVAL_LANGUAGE"
echo "  Source Langauge:    $SOURCE_LANGUAGE"
echo "  Experimental Setting:$EXP"
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


if [[ "$EXP" == "translation"* ]]; then
    IFS=',' read -ra SOURCE_LANGUAGES <<< "$SOURCE_LANGUAGE"
    echo "With translation from languages: ${SOURCE_LANGUAGES[*]}"
    
    # Use a loop to build the JSON array safely
    SOURCE_LANGUAGES_STRING="["
    for i in "${!SOURCE_LANGUAGES[@]}"; do
        SOURCE_LANGUAGES_STRING+="\"${SOURCE_LANGUAGES[$i]}\""
        # Add comma if not the last element
        if [ $i -lt $((${#SOURCE_LANGUAGES[@]} - 1)) ]; then
            SOURCE_LANGUAGES_STRING+=", "
        fi
    done
    SOURCE_LANGUAGES_STRING+="]"
fi


python3 "main.py" \
--hf_token="$HF_TOKEN" \
--hf_dataset_path="$HF_DATASET_PATH" \
--model_id "$MODEL_ID" \
--sample_size="$SAMPLE_SIZE" \
--eval_languages="$EVAL_LANGUAGES_STRING" \
--source_languages="$SOURCE_LANGUAGES_STRING" \
--results_folder="$RESULTS_FOLDER" \
--exp="$EXP" \
--target_stereotype="$TARGET_STEREOTYPE_STRING" \

# RESULTS_DIR="/Users/s2583833/Downloads/EURO_LLM_9B_I" 
# RESULTS_DIR="/Users/s2583833/Downloads/EURO_LLM_9B_I_tasks"

# python3 "$REPO_SUBDIR/compare_results_eng.py" \
# --model_id "$MODEL_ID" \
# --model_label "$MODEL_LABEL" \
# --results_dir="$RESULTS_DIR" \
# --languages="$EVAL_LANGUAGES_STRING" \


# rsync -avh "$RESULTS_FOLDER/" "$HOME/results"

# Deactivate the environment when finished
# deactivate
