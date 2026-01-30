#!/bin/bash

# Add slurm directives and set up environment / modules as necessary if running on a cluster... 

./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "none" "translation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "instruction" "translation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "none" "translation_MCQ" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "instruction" "translation_MCQ" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "none" "translation_open" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "instruction" "translation_open" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "none" "generation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "instruction" "generation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "none" "generation_MCQ" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "Spanish" "English" "instruction" "generation_MCQ" 

./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "English" "English" "none" "generation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "English" "English" "instruction" "generation" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "English" "English" "none" "generation_MCQ" 
./model_eval.sh "utter-project/EuroLLM-1.7B-Instruct" "EURO_LLM_1.7B_I" 100 "English" "English" "instruction" "generation_MCQ" 
