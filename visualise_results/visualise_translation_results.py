import ast
import os
from pathlib import Path
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

DEFAULT_RESULTS_FOLDER = Path.home() / "Desktop" / "translation_initial_results"
results_folder = Path(os.getenv("EUROGEST_RESULTS_FOLDER", DEFAULT_RESULTS_FOLDER))


# EURO_LLM_LANGS = {
#     'Bulgarian': 'bg', 'Catalan': 'ca', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 
#     'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 
#     'Galician': 'gl', 'German': 'de', 'Greek': 'el', 'Hungarian': 'hu', 'Irish': 'ga', 
#     'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt', 'Norwegian': 'no', 
#     'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 
#     'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk'
# }

EURO_LLM_LANGS = {"French": "fr", "German": "de", "Italian": "it", "Spanish": "es", "Polish": "pl"}

def load_results(lang):
    file_path = f'results_outputs/{lang}_results.json'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None 
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the two main parts
    logs_summary = data.get('logs_summaries', [])
    gen_summary = data.get('generated_summaries', {})
    
    # Optional: Convert prompt_id keys back to their original type if needed
    # (e.g., converting "1" back to 1)
    formatted_gen_summary = {}
    for p_id_str, counts in gen_summary.items():
        # If your prompt IDs were originally integers:
        try:
            p_id = int(p_id_str)
        except ValueError:
            p_id = p_id_str # Keep as string if it's "2a", "4_MCQ", etc.
            
        formatted_gen_summary[p_id] = counts

    return logs_summary, formatted_gen_summary
  
def calculate_generation_ratios_with_inv(counts):

    total = sum(counts.values())
    masc_rate = (counts.get('m', 0) / total) if total > 0 else 0.0
    fem_rate = (counts.get('f', 0) / total) if total > 0 else 0.0
    neutral_rate = (counts.get('n', 0) / total) if total > 0 else 0.0

    return masc_rate, fem_rate, neutral_rate

def calculate_generation_ratios_without_inv(counts):

    total = sum(counts.values()) - counts.get('invalid', 0)
    masc_rate = (counts.get('m', 0) / total) if total > 0 else 0.0
    fem_rate = (counts.get('f', 0) / total) if total > 0 else 0.0
    neutral_rate = (counts.get('n', 0) / total) if total > 0 else 0.0

    return masc_rate, fem_rate, neutral_rate

def init_context_counts():
    return {"m": 0, "f": 0, "n": 0, "invalid": 0}

# initialise structures for plotting

prompt_ids_order = ['baseline','1','2','3','4_MCQ','4a_MCQ', '5_MCQ','5a_MCQ','6_MCQ','6a_MCQ']

# load results from file 
results_generated_summaries = {}
results_log_summaries = {}

for lang in EURO_LLM_LANGS.keys():

    logs_results, gen_summaries = load_results(lang)
    if logs_results is None:
        continue

    results_log_summaries[lang] = logs_results
    results_generated_summaries[lang] = gen_summaries

def plot_generation_ratio_data(results_generated_summaries, condition="rmasc"):
    # Label mapping for the X-axis
    prompt_labels = {
        '1': '1: Tag Only',
        '2': '2: Instruction',
        '3': '3: Missing Word',
        '4_MCQ': '4: Missing Word MCQ (masc first)',
        '4a_MCQ': '4a: Missing Word MCQ (fem first)',
        '5_MCQ': '5: Choose Best (masc first)',
        '5a_MCQ': '5a: Choose Best (fem first)',
        '6_MCQ': '6: Choose Best (with Neut)',
        '6a_MCQ': '6a: Choose Best (with Neut, fem first)',
        'baseline': 'Baseline'
    }

    plt.figure(figsize=(14, 8))
    colors = cm.get_cmap('tab10', len(results_generated_summaries))

    for i, (lang, gen_summaries) in enumerate(results_generated_summaries.items()):
        current_color = colors(i % 10)
        
        p_ids_to_plot = []
        masc_rates_fem_context = []
        masc_rates_masc_context = []

        for prompt_id in prompt_ids_order:
            summary = gen_summaries.get(prompt_id)
            if not summary:
                print(prompt_id, lang, "No summary found, inserting null values")
                fem_counts = {'masculine': 0, 'feminine': 0, 'neutral': 0}
                masc_counts = {'masculine': 0, 'feminine': 0, 'neutral': 0}
            else:
                fem_counts = summary.get('fem_context_counts', {'masculine': 0, 'feminine': 0, 'neutral': 0})
                masc_counts = summary.get('masc_context_counts', {'masculine': 0, 'feminine': 0, 'neutral': 0})

            # Calculate ratio: Masc / (Masc + Fem)
            # We ignore neutral for the binary ratio to match the "Masc Rate" logic
            masc_rate_m, fem_rate_m, neutral_rate_m = calculate_generation_ratios_with_inv(masc_counts)
            masc_rate_f, fem_rate_f, neutral_rate_f = calculate_generation_ratios_with_inv(fem_counts)

            p_ids_to_plot.append(prompt_labels.get(str(prompt_id), str(prompt_id)))
            masc_rates_fem_context.append(masc_rate_f)
            masc_rates_masc_context.append(masc_rate_m)

        if condition == "rmasc":
            # Plot Masculine Rate in Female-Stereotyped contexts (Solid)
            plt.plot(p_ids_to_plot, masc_rates_fem_context, 
                    marker='o', label=f"{lang} (Fem-Stereo Context)", 
                    color=current_color, linewidth=2)
            
            # Plot Masculine Rate in Masculine-Stereotyped contexts (Dashed)
            plt.plot(p_ids_to_plot, masc_rates_masc_context, 
                    marker='s', label=f"{lang} (Masc-Stereo Context)", 
                    color=current_color, linestyle='--', linewidth=2)
            
        elif condition == "gs":
            # Plot GS Score for each language (Masc / (Masc + Fem))
            gs_scores = [m / (f) if (f) > 0 else 1.0 for m, f in zip(masc_rates_masc_context, masc_rates_fem_context)]
            plt.plot(p_ids_to_plot, gs_scores, 
                    marker='^', label=f"{lang} GS Score", 
                    color=current_color, linewidth=2)

    if condition == "gs":
        plt.ylabel('GS Score', fontweight='bold')
        plt.title('Gender Stereotyping in generation across Prompt IDs', fontsize=14, fontweight='bold')
        plt.ylabel('gs scores', fontweight='bold')
        plt.ylim(0.8, 2.0) # Adjusted range for GS scores
        plt.axhline(1.0, color='black', linestyle='-', alpha=0.3, label='Neutral (1.0)')

    elif condition == "rmasc":
    # Styling (Matching your logprobs plot)
        plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label='Neutral (0.5)')
        plt.title('Generation Bias: Masculine Output Rate by Prompt Strategy', fontsize=14, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.ylabel('Masculine / Feminine generation ratio', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Prompt Condition', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_generation_bar_data(results_generated_summaries):
    prompt_labels = {'4_MCQ': '4: Missing Word MCQ (masc first)',
        '4a_MCQ': '4a: Missing Word MCQ (fem first)',
        '5_MCQ': '5: Choose Best (masc first)',
        '5a_MCQ': '5a: Choose Best (fem first)',
        '6_MCQ': '6: Choose Best (with Neut)',
        '6a_MCQ': '6a: Choose Best (with Neut, fem first)',
        'baseline': 'Baseline'}
    
    # Filter order and identify labels
    filtered_order = [p for p in prompt_ids_order if p in prompt_labels.keys()]
    display_labels = [prompt_labels.get(str(p), str(p)) for p in filtered_order]
    
    n_langs = len(results_generated_summaries)
    fig, axes = plt.subplots(n_langs, 1, figsize=(14, 5 * n_langs), constrained_layout=True)
    
    if n_langs == 1:
        axes = [axes]

    for ax, (lang, gen_summaries) in zip(axes, results_generated_summaries.items()):
        # Data containers for the bars
        m_rates, f_rates, n_rates = [], [], []

        for p_id in filtered_order:
            summary = gen_summaries.get(p_id)
            if not summary:
                m_rates.append(0); f_rates.append(0); n_rates.append(0)
                continue
            
            # Aggregate counts across both Fem and Masc contexts for a general overview
            # Or you can choose to plot only one context type
            f_counts = summary.get('fem_context_counts', init_context_counts())
            m_counts = summary.get('masc_context_counts', init_context_counts())
            
            # Combine counts for the prompt overall
            total_counts = {k: f_counts.get(k, 0) + m_counts.get(k, 0) for k in ['m', 'f', 'n', 'invalid']}
            
            m_r, f_r, n_r = calculate_generation_ratios_with_inv(total_counts)
            m_rates.append(m_r)
            f_rates.append(f_r)
            n_rates.append(n_r)

        # --- Bar Geometry ---
        x = np.arange(len(display_labels))
        width = 0.25  # Width of individual bars

        ax.bar(x - width, m_rates, width, label='Masculine', color='#3498db', edgecolor='black', alpha=0.8)
        ax.bar(x, f_rates, width, label='Feminine', color='#e74c3c', edgecolor='black', alpha=0.8)
        ax.bar(x + width, n_rates, width, label='Neutral', color='#bdc3c7', edgecolor='black', alpha=0.8)

        # --- Formatting ---
        ax.set_title(f'Generation Distribution: {lang}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=30, ha='right')
        ax.set_ylabel('Ratio of Total Generations')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize='small')

    plt.suptitle('Gender Generation Ratios per Prompt Strategy', fontsize=18, fontweight='bold', y=1.02)
    plt.show()




def plot_logprobs_data(results_log_summaries, condition="rmasc"): 

    prompt_labels = {
        '1': '1: Tag Only',
        '2': '2: Instruction',
        '3': '3: Missing Word',
        '4_MCQ': '4: Missing Word MCQ (masc first)',
        '4a_MCQ': '4a: Missing Word MCQ (fem first)',
        '5_MCQ': '5: Choose Best (masc first)',
        '5a_MCQ': '5a: Choose Best (fem first)',
        '6_MCQ': '6: Choose Best (with Neut)',
        '6a_MCQ': '6a: Choose Best (with Neut, fem first)',
        'baseline': 'Baseline'
    }
    plt.figure(figsize=(12, 7))
    colors = cm.get_cmap('tab10', len(results_log_summaries))
    
    for i, (lang, summary_list) in enumerate(results_log_summaries.items()):
        # 1. Create the DataFrame once per language
        lang_df = pd.DataFrame(summary_list)
        
        # Ensure prompt_id is a string to match prompt_ids_order
        lang_df['prompt_id'] = lang_df['prompt_id'].astype(str)
        
        # 2. Sort the data based on your specific prompt order
        lang_df['prompt_id'] = pd.Categorical(lang_df['prompt_id'], categories=prompt_ids_order, ordered=True)
        lang_df = lang_df.sort_values('prompt_id')

        display_labels = [prompt_labels.get(pid, pid) for pid in lang_df['prompt_id']]
        current_color = colors(i)
        
        if condition == "gs":
            # GS Score = Masc / Fem) for each prompt condition
            print(lang_df)
            lang_df["gs_score"] = lang_df["masc_context_bias"] / lang_df["fem_context_bias"].fillna(0.5)      
            print(lang_df)
            plt.plot(display_labels, lang_df['gs_score'], 
                    marker='o', label=f"{lang} GS Score", 
                    color=current_color, linewidth=2)
        else:
            # Plot Masculine Rate in Female contexts
            plt.plot(display_labels, lang_df['fem_context_bias'], 
                    marker='o', label=f"{lang} (Fem-Stereo)", 
                    color=current_color, linewidth=2)
            
            # Plot Masculine Rate in Masculine contexts
            plt.plot(display_labels, lang_df['masc_context_bias'], 
                    marker='s', label=f"{lang} (Masc-Stereo)", 
                    color=current_color, linestyle='--', linewidth=2)

    # Dynamic styling based on condition
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability

    if condition == "gs":
        plt.ylabel('GS Score', fontweight='bold')
        plt.title('Gender Stereotype (GS) Scores across Prompt IDs', fontsize=14, fontweight='bold')
        plt.ylim(0.8, 2.0) # Adjusted range for GS scores
        plt.axhline(1.0, color='black', linestyle='-', alpha=0.3, label='Neutral (1.0)')
    else:
        plt.ylabel('Avg Masculine Probability Rate', fontweight='bold')
        plt.title('Masculine Rate Comparison Across Prompt IDs', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label='Neutral (0.5)')
    
    plt.xlabel('Prompt ID', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# plot_generation_ratio_data(results_generated_summaries, condition="gs")
plot_generation_bar_data(results_generated_summaries)

plot_logprobs_data(results_log_summaries, condition="gs") # condition can be rmasc or gs

