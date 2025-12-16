#!/bin/bash

# activate virtual environment if needed

source venv/bin/activate
# ========================================= #
# ======== TRANSLATE WITH GOOGLE API======= #
# ========================================= #

# where the dataset for translation is saved 
# export DATA_DIR="path/to/data/to/translate.csv"
export DATA_DIR="./gest_1.1.csv"

# Where to save the raw translations
# export RAW_TRANSLATIONS_DIR="/path/to/raw/translations"
export RAW_TRANSLATIONS_DIR="$HOME/Desktop/251215_testing"

# Google API key JSON and project name
# export JSON_KEY="/path/to/your/key"
# export PROJECT_ID="your_project_key"
export JSON_KEY="/Users/s2583833/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD/Research/EuroGEST/keys/gest-449016-bcaa9e77f7b4.json"
export PROJECT_ID="gest-449016"

# Sample size: 1 = full dataset, other n = test n sentences (££)
export SAMPLE_SIZE=5

# python translation_script.py

# ========================================= #
# ======== FILTER WITH COMET QE =========== #
# ========================================= #

# where to save the quality-filtered translations
# export FILTERED_TRANSLATIONS_DIR="/path/to/filtered/translations"
export FILTERED_TRANSLATIONS_DIR="$HOME/Desktop/251215_filtering"
export THRESHOLD="0.86"
export QE_MODEL="Unbabel/wmt22-cometkiwi-da"

# python qe_filtering.py


# ========================================= #
# ======== CREATE FINAL DATASET =========== #
# ========================================= #

# where to save the quality-filtered translations
# export FINAL_DATA_DIR="/path/to/final/dataset"
export FINAL_DATA_DIR="$HOME/Desktop/251215_finished"

# set heuristic filtering parameters for detecting gendered minimal pairs (e.g. 2 letters on 1 word)
export NUM_GENDERED_WORDS="2"
export NUM_DIFFERENT_LETTERS="2"

python heuristic_filtering.py
