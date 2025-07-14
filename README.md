# EuroGEST Dataset

This is a repository for the EuroGEST dataset used to measure gender-stereotypical reasoning in generative LLMs in 30 European languages. 

- [Arxiv Pre-print](https://arxiv.org/abs/2506.03867)
  
- The original GEST dataset, which this work builds on, is from [Women Are Beautiful, Men Are Leaders: Gender Stereotypes in Machine Translation and Language Modeling](https://arxiv.org/abs/2311.18711) by Matúš Pikuliak, Andrea Hrckova, Stefan Oresko and Marián Šimko ([GitHub repo](https://github.com/kinit-sk/gest/tree/main?tab=readme-ov-file)). 

## Changelog



## Data

The data folder includes two csv files with translations of all of the first-person singular English sentences from the original GEST dataset into 29 additional European languages:
1. Gender insensitive translations, where the sentence is expressed the same regardless of whether the speaker is a man or a woman. These sentences require wrapping in a gendered template (e.g. '"s", he/she said') before they can be used to probe for gender bias. 
2. Gender sensitive translations, where the gender of the speaker is expressed morphologically on the sentence. This file includes two columns per language, one for masculine variants and one for feminine variants, which can be used to probe for gender bias directly. 

The folder also includes a csv file listing all included languages and the numbers of gender sensitive and gender insensitive sentences available for each language. 

Code for translation of GEST dataset to be released shortly 


## Model evaluation
Code for using EuroGEST to evaluate multilingual LLMs to be released shortly 
