#!/bin/bash

source .venv/bin/activate

## Paths
RESULTS_DIR="../model_evaluation_results"
GRAPHS_DIR="../model_evaluation_graphs"

SUBSET="all" # options "all" "EU" langs only
SORT_BY="family" # options = 'family', 'alphabetical' (order in which languages are displayed)

# Check if Results Directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory $RESULTS_DIR does not exist."
    exit 1
fi

# 1. Find all subdirectories in the results folder to get model labels
# We use 'basename' to get just the folder name, not the full path
MODEL_LABELS=()
for dir in "$RESULTS_DIR"/*/; do
    if [ -d "$dir" ]; then
        MODEL_LABELS+=("$(basename "$dir")")
    fi
done

# 2. Check if we found any models
if [ ${#MODEL_LABELS[@]} -eq 0 ]; then
    echo "No model results found in $RESULTS_DIR"
    exit 1
fi

echo "Found models: ${MODEL_LABELS[*]}"

LABELS_STRING=$(printf '"%s", ' "${MODEL_LABELS[@]}")
LABELS_STRING="[${LABELS_STRING%, }]"

LANGUAGES=(
    "Bulgarian" "Catalan" "Croatian" "Czech" "Danish" "Dutch" 
    "English" "Estonian" "Finnish" "French" "Galician" "German" 
    "Greek" "Hungarian" "Irish" "Italian" "Latvian" "Lithuanian" 
    "Maltese" "Norwegian" "Polish" "Portuguese" "Romanian" "Russian" 
    "Slovak" "Slovenian" "Spanish" "Swedish" "Turkish" "Ukrainian"
)

# Construct the JSON-like string for the Python argument
LANGUAGES_STRING=$(printf '"%s", ' "${LANGUAGES[@]}")
LANGUAGES_STRING="[${LANGUAGES_STRING%,}]"  

python3 visualise_results.py \
--model_labels="$LABELS_STRING" \
--languages="$LANGUAGES_STRING" \
--results_dir="$RESULTS_DIR" \
--graphs_folder="$GRAPHS_DIR" \
--subset="$SUBSET" \
--sort_by="$SORT_BY" 