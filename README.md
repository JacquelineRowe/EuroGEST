# EuroGEST Dataset

This is the GitHub repository for the paper [EuroGEST: Investigating gender stereotypes in multilingual language models](https://aclanthology.org/2025.emnlp-main.1632/).* Our dataset and dataset card is now hosted on [HuggingFace](https://huggingface.co/datasets/utter-project/EuroGEST).

This repository includes:
1. Dataset Creation
   - Scripts used to translate gendered minimal pairs of the GEST dataset from English into 29 European langauges using the Google Translate API, creating EuroGEST
   - Basic statistics about the EuroGEST dataset created using this method
2. Model Evaluation
   - Scripts for evaluating any auto-regressive LLM for gendered stereotyping using the EuroGEST dataset (which is hosted on hugging face).
   - This folder includes json files for the punctuation and prompt scaffolds used for evaluating gender neutral sentences across languages 
3. Results visualisation
   - Scripts for comparing and visualising gendered stereotyping across different models, generating:
     - heatmaps of stereotype rankings or intensities, both per language and over all languages
     - visualisation of overall g_s scores, which measures the model prefers stereotypical sentence genders over antistereotypical ones.

In each folder, we include a read.me with details about how to setup and run the code. 

*Jacqueline Rowe, Mateusz Klimaszewski, Liane Guillou, Shannon Vallor, and Alexandra Birch. 2025. EuroGEST: Investigating gender stereotypes in multilingual language models. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_, pages 32062–32084, Suzhou, China. Association for Computational Linguistics.

# GEST Dataset
The original GEST dataset, which this work builds on, is from [Women Are Beautiful, Men Are Leaders: Gender Stereotypes in Machine Translation and Language Modeling](https://arxiv.org/abs/2311.18711) by Matúš Pikuliak, Andrea Hrckova, Stefan Oresko and Marián Šimko ([GitHub repo](https://github.com/kinit-sk/gest/tree/main?tab=readme-ov-file)). 


