#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=120g
#SBATCH --job-name=model_eval
#SBATCH --account=plgmodularnlp-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

ml ML-bundle/24.06a

export HF_HOME=/net/storage/pr3/plgrid/plggmultilingualnlp/huggingface
mkdir -p $HF_HOME

cd "$PLG_GROUPS_STORAGE/plggmultilingualnlp/genderbias"
source venv/bin/activate

HUGGINGFACE_HUB_TOKEN=os.getenv("HF_TOKEN") # set from env 
GIT_REPO="$PLG_GROUPS_STORAGE/plggmultilingualnlp/EuroGEST" 

# Usage check
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <MODEL_ID> <MODEL_LABEL> <EXP_ID> <SAMPLE_SIZE>"
  echo "Example: $0 utter-project/EuroLLM-1.7B-Instruct EURO_LLM_1.7B_I 14 1"
  exit 1
fi

MODEL_ID=$1
MODEL_LABEL=$2
EXP_=$3
SAMPLE_SIZE=$4 # 1 is entire set, otherwise use smaller values for testing 

# specify languages here not in command line 
LANGUAGES=("English" "Bulgarian" "Danish" "Dutch" "Estonian" \
        "Finnish" "French" "German" "Greek" "Hungarian" "Irish" \
        "Italian" "Latvian" "Lithuanian" "Maltese" "Portuguese" "Romanian" \
        "Spanish" "Swedish" "Catalan" "Galician" "Norwegian" "Turkish" "Croatian" \
        "Czech" "Polish" "Slovak" "Slovenian" "Russian" "Ukrainian") # add more if needed

LANGUAGES_STRING=$(printf '"%s", ' "${LANGUAGES[@]}")
LANGUAGES_STRING="[${LANGUAGES_STRING%,}]"
        
RESULTS_FOLDER=f"results/{$MODEL_LABEL}/{$EXP_ID}"

python3 model_eval.py \
--hf_token="$HUGGINGFACE_HUB_TOKEN" \
--model_id "$MODEL_ID" \
--model_label "$MODEL_LABEL" \
--git_repo_path="$GIT_REPO" \
--sample_size="$SAMPLE_SIZE" \
--languages="$LANGUAGES_STRING" \
--results_folder="$RESULTS_FOLDER"




