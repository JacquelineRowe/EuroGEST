
import pandas as pd
import os
from pathlib import Path


gest_original_df = pd.read_csv("create_dataset/gest_1.1.csv")

# --- CONFIGURATION ---
# Define the languages and stereotype IDs
NEUTRAL_LANGS = ['Danish', 'Dutch', 'Estonian', 'Finnish', 'Hungarian', 'Irish', 'Swedish', 'Norwegian', 'Turkish']
STEREOTYPES = range(1, 17) # Your full range: 1, 2, 3, ..., 16

# Set the input and output directories (assuming they are the same folder where you run the script)
INPUT_DIR = Path('/Users/s2583833/Library/CloudStorage/OneDrive-UniversityofEdinburgh/GitRepos/genderbias/dataset_creation/2_quality_filtered_translations/0.85')
OUT_DIR = Path('/Users/s2583833/Desktop/251216_temp')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME_TEMPLATE_1 = "neutral_stereotype_{stereotype_id}_filtered.csv"
FILE_NAME_TEMPALTE_2 = "gendered_stereotype_{stereotype_id}_filtered.csv"



def recompile_language_data(language: str, stereotypes: range, input_dir: Path, out_dir: Path, 
                             file_name_template: str, gest_stereotype_map: pd.DataFrame) -> None:
    """
    Reads all stereotype CSV files, extracts data for a specific language, 
    consolidates it, and merges the GEST stereotype ID.

    ... [Arguments are the same] ...
    """
    all_stereotype_dfs = []
    
    # 1. Iterate through all stereotype files
    for stereotype_id in stereotypes:
        file_name = file_name_template.format(stereotype_id=stereotype_id)
        file_path = input_dir / file_name

        if not file_path.exists():
            continue

        try:
            # Read the CSV, using the GEST ID (the first column) as the index
            df = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            print(f"Error reading {file_name}: {e}. Skipping.")
            continue

        # 2. Select the necessary columns: GEST_sentence and the language translation
        required_cols = ['GEST_sentence', language]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            continue

        df_subset = df[required_cols].copy()
        
        # Rename the translation column (e.g., 'Danish' -> 'Danish_translation')
        df_subset.rename(columns={language: f'{language}_translation'}, inplace=True)
        
        all_stereotype_dfs.append(df_subset)

    if not all_stereotype_dfs:
        print(f"No usable stereotype data found for {language}. Skipping compilation.")
        return

    # 3. Concatenate all dataframes: Stacks the data and aligns rows by the GEST ID index.
    final_df = pd.concat(all_stereotype_dfs, axis=0)
    
    # 4. Consolidate and Merge Stereotype ID
    final_df = final_df.groupby(final_df.index).first()
    
    # Merge the compiled data with the original stereotype ID based on the GEST ID index
    final_df = final_df.merge(
        gest_stereotype_map[['stereotype']], 
        left_index=True, 
        right_index=True, 
        how='left' # Keep all compiled sentences
    )
    
    # Reorder columns to put stereotype_id first
    cols = ['stereotype', 'GEST_sentence'] + [col for col in final_df.columns if col not in ['stereotype', 'GEST_sentence']]
    final_df = final_df[cols]
    
    # 5. Save the final DataFrame
    output_file_name = f"{language.lower().replace(' ', '_')}_compiled.csv"
    final_df.to_csv(out_dir / output_file_name)

   # Make sure 'GEST_ID' is the column that matches the index of the other CSVs.
gest_stereotype_map = gest_original_df[['stereotype']]

for lang in NEUTRAL_LANGS:
    recompile_language_data(
        language=lang, 
        stereotypes=STEREOTYPES, 
        input_dir=INPUT_DIR, 
        out_dir=OUT_DIR,
        file_name_template=FILE_NAME_TEMPLATE_1,
        gest_stereotype_map=gest_stereotype_map # Pass the map here
    )


GENDERED_LANGS = {
    'Bulgarian': 'bg', 'French': 'fr', 'German': 'de', 'Greek': 'el',
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Spanish': 'es', 'Catalan': 'ca',
    'Galician': 'gl', 'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Russian': 'ru', 'Ukrainian': 'uk'
}

NEUTRAL_LANGS = {
    'Danish': 'da', 'Dutch': 'nl', 'Estonian': 'et', 'Finnish': 'fi',
    'Hungarian': 'hu', 'Irish': 'ga', 'Swedish': 'sv',
    'Norwegian': 'no', 'Turkish': 'tr'
}



gender_insensitive_path = "/Users/s2583833/Library/CloudStorage/OneDrive-UniversityofEdinburgh/GitRepos/genderbias/dataset_creation/all_gender_insensitive.csv"
gender_sensitive_path = "/Users/s2583833/Library/CloudStorage/OneDrive-UniversityofEdinburgh/GitRepos/genderbias/dataset_creation/all_gender_sensitive.csv"

gender_insensitive_data = pd.read_csv(gender_insensitive_path)
gender_sensitive_data = pd.read_csv(gender_sensitive_path, header=[0,1])  

print("Available Keys/Levels:", gender_sensitive_data.columns.get_level_values(1).unique())

for lang in GENDERED_LANGS.keys():

    language_data = pd.DataFrame(columns=["GEST sentence", 
                                 "neutral translation", 
                                 "masculine translation", 
                                 "feminine translation", 
                                 "gendered word masculine",
                                 "gendered word feminine",
                                 "original_stereotype"])
    
    language_data["GEST sentence"] = gest_original_df["sentence"]
    language_data["original_stereotype"] = gest_original_df["stereotype"]


    lang_data_masc = gender_sensitive_data[lang, "the man said"]
    lang_data_fem = gender_sensitive_data[lang, "the woman said"]
    lang_data_neutral = gender_insensitive_data[lang]

    # take the rows for this language from this dataset and put into a language-specific df to match new format 


for lang in NEUTRAL_LANGS.keys():
    lang_data_neutral = gender_insensitive_data[lang]






