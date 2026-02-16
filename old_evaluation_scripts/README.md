# EuroGEST: Model Evaluation

This sub-directory contains the scripts and configurations required to evaluate autoregressive LLMs using the EuroGEST dataset downloaded from Hugging Face. The evaluation is designed to be highly configurable via the shell script.

These scripts measure the log-likelihood of the masculine and feminine versions of each sentence available in each EuroGEST-supported language. Each sentence relates to a stereotype about men and women. For grammatically gender-neutral sentences, to measure gendered stereotyping we wrap the sentence in a masculine or a feminine template (both pronouns, e.g. he/she said X, and nouns, e.g. the man/woman said X, are measured. These are stored in the prompt_scaffolds.json file for each language.

The results are saved into .csv files per language, which include the log likelihoods and relative probability of the masculine version of each sentence. These results can be processed using the scripts in the 'visualise results' subfolder of the main github repository. 

## Setup Instructions

**1. Setup Environment or load docker container**

Add relevant cluster configuration and environment setup to the top of your .sh script. 

If using a venv: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```
Note: Evaluation typically requires a GPU with sufficient VRAM (e.g., NVIDIA A100 or Apple Silicon with MPS) depending on the model size.

**2. Configure the Evaluation**

Set desired parameters in evaluate_models.sh:
- MODEL_ID: The HuggingFace path or local path to the model.
- MODEL_LABEL: A shorthand label for that model for logging 
- LANGUAGES: A comma-separated list of languages to test, or "all" for all in the set. 
- SAMPLE_SIZE - how many samples you want to test (useful for testing and debugging) 

Set the path where you want to save the results. 

**If evaluating gated models (like Llama-3), ensure you are logged into HuggingFace via huggingface-cli login or have your HF_TOKEN exported in your environment.**

**4. Execute**
```bash
./evaluate_models.sh
```
