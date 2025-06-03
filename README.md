# EuroGEST Dataset

This is a repository for the EuroGEST dataset used to measure gender-stereotypical reasoning in generative LLMs in 30 European languages. 

- Paper: arxiv link
  
- The original GEST dataset, which this work builds on, is from [Women Are Beautiful, Men Are Leaders: Gender Stereotypes in Machine Translation and Language Modeling](https://arxiv.org/abs/2311.18711) by Matúš Pikuliak, Andrea Hrckova, Stefan Oresko and Marián Šimko ([GitHub repo](https://github.com/kinit-sk/gest/tree/main?tab=readme-ov-file)). 

## Changelog



## Data

The data folder includes two csv files with translations of all of the first-person singular English sentences from the original GEST dataset into 29 additional European languages. One csv file includes gender insensitive translations (where the sentence is expressed the same regardless of whether the speaker is a man or a woman) and the other includes gender sensitive translations (where the gender of the speaker is expressed morphologically on the sentence). The gender-sensitive csv includes two columns per language, one for masculine variants and one for feminine variants. 

The folder also includes a list of all included languages and the numbers of gender sensitive and gender insensitive sentences available for each. 

Code for translation of GEST dataset to be released shortly 


## Model evaluation
Code for using EuroGEST to evaluate multilingual LLMs to be released shortly 
