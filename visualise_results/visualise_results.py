#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:06:26 2025

"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import fire
import os
import re
import matplotlib.style as style 

style.use('tableau-colorblind10')
mpl.rcParams['font.family'] = 'Times New Roman'

EURO_LLM_LANGS = {
    'Bulgarian': 'bg', 'Catalan': 'ca', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 
    'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 
    'Galician': 'gl', 'German': 'de', 'Greek': 'el', 'Hungarian': 'hu', 'Irish': 'ga', 
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt', 'Norwegian': 'no', 
    'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 
    'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk'
}


EU_LANGS = {
    'Bulgarian': 'bg', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 
    'English': 'en', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'German': 'de', 
    'Greek': 'el', 'Hungarian': 'hu', 'Irish': 'ga', 'Italian': 'it', 'Latvian': 'lv', 
    'Lithuanian': 'lt', 'Maltese': 'mt', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 
    'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv'
}

LANGS_SORTED_BY_FAMILY = {'Bulgarian': 'bg', 'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl','Russian': 'ru', 
                           'Slovak': 'sk', 'Slovenian': 'sl', 'Ukrainian': 'uk',

                        'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'German': 'de', 'Norwegian': 'no','Swedish': 'sv', 

                        'Catalan': 'ca', 'French': 'fr', 'Galician': 'gl', 'Italian': 'it', 'Portuguese': 'pt', 
                        'Romanian': 'ro',  'Spanish': 'es', 
                          
                        'Latvian': 'lv', 'Lithuanian': 'lt', 

                        'Estonian': 'et', 'Finnish': 'fi', 'Hungarian': 'hu',
                          
                        'Greek': 'el',  'Irish': 'ga', 'Maltese': 'mt','Turkish': 'tr', }


ALL_LANGS = dict(sorted(EURO_LLM_LANGS.items()))
EU_LANGS = dict(sorted(EU_LANGS.items()))

STEREOTYPE_LABELS = {
    1: "Emotional", 2: "Gentle", 3: "Empathetic", 4: "Neat", 5: "Social", 
    6: "Weak", 7: "Beautiful", 8: "Tough", 9: "Self-confident", 10: "Professional", 
    11: "Rational", 12: "Providers", 13: "Leaders", 14: "Childish", 15: "Sexual", 16: "Strong"
}

def plot_stereotype_positions(df, languages, model_label, graphs_dir, sort_by):
    graphs_folder = os.path.join(graphs_dir, 'stereotype_positions')
    os.makedirs(graphs_folder, exist_ok=True)
    
    available_langs = [l for l in languages if l in df.columns]

    ordered_stereotype_labels = list(STEREOTYPE_LABELS.values())
    heatmap_df = pd.DataFrame(index=ordered_stereotype_labels, columns=available_langs)

    ## 3. Fill the heatmap with the ranks from the original DF
    for lang in available_langs:
        for rank, stereotype in df[lang].items():
            if stereotype in heatmap_df.index:
                # Add +1 to convert 0-15 into 1-16
                heatmap_df.loc[stereotype, lang] = rank + 1

    # Convert to float for Seaborn compatibility
    heatmap_df = heatmap_df.astype(float)

    # 3. Plotting
    plt.figure(figsize=(10, 10)) # Vertical orientation needs more height
    
    # cmap='PuOr_r' will color based on the rank (0-15)
    ax = sns.heatmap(heatmap_df, 
                     cmap='PuOr_r', 
                     square=True, 
                     center=7.5, 
                     vmin=1, 
                     vmax=16, 
                     annot=True, # Optional: shows the rank number in the box
                     fmt=".0f",
                     cbar_kws={'label': 'Masculine Rank (1=Most Masculine)'})
    
    plt.xlabel('Language', fontsize=14, labelpad=10)
    plt.ylabel('Stereotype', fontsize=14, labelpad=10)

    try:
        lang_codes = [EURO_LLM_LANGS[lang].upper() for lang in heatmap_df.columns]
        ax.set_xticklabels(lang_codes, ha='center', fontsize=10)
    except NameError:
        ax.set_xticklabels(heatmap_df.columns, ha='center', fontsize=10)
    
    # Y-axis labels are now the Stereotype names
    ax.set_yticklabels(heatmap_df.index, rotation=0, fontsize=10)

    plt.tight_layout()
    
    # Save logic
    safe_label = str(model_label).replace("/", "_")
    save_path = os.path.join(graphs_folder, f'{safe_label}_{sort_by}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Stereotype position heatmap saved to: {save_path}")
    plt.close()
    
def parse_model_label(label):
    # 1. Extract Size (e.g., 1.7)
        size_match = re.search(r"(\d+\.?\d*)B", label)
        size = size_match.group(1) if size_match else "Unknown"
        
        # 2. Extract Name (Everything before the size or first underscore)
        # This splits by _ or digits and takes the first part
        name_match = re.split(r'_|\d', label)[0]
        
        # 3. Check for Instruct
        is_instruct = label.endswith(('_I', '_Instruct', '_i', '_instruct'))
        
        return (
            name_match,
            size,
            is_instruct
        )
    


def plot_stereotype_rates_per_lang(model_labels, graphs_dir, langs, dfs, subset):
    # 1. Setup language Grid
    num_rows = 6 if subset != "EU" else 5
    num_cols = 5

    fig, axes = plt.subplots(num_rows,num_cols,figsize=(25,35)) # Create a single subplot
    axes = axes.flatten()

    graphs_folder = os.path.join(graphs_dir, 'stereotype_rates')
    os.makedirs(graphs_folder, exist_ok=True)
    
    model_family_map = {}
    marker_list = ['o', '*', 'o', '^', '*', 'o']  
    color_palette = ['#F0A050', '#AB5C8E', '#4682B4', '#5D9C59', 
              '#E6794C', '#8B008B', '#2F4F4F', '#FF69B4', '#8A2BE2']
    
    for model_label in model_labels:
        df=dfs[model_label]
        m_name, model_size, is_instruct = parse_model_label(model_label)

        # set family index for colour / shape 
        if m_name not in model_family_map:
            # We multiply by 2 so each family jumps ahead by two slots
            model_family_map[m_name] = (len(model_family_map) * 2) % len(color_palette)
            
        family_base_idx = model_family_map[m_name]
        final_idx = (family_base_idx + 1) % len(color_palette) if is_instruct else family_base_idx

        color = color_palette[final_idx]
        marker = marker_list[final_idx % len(marker_list)]

        for lang_index, language in enumerate(langs):
            ax = axes[lang_index]

            lang_row = df[df['Unnamed: 0'] == language]

            ax.set_ylabel('Stereotype rate', fontsize=16)
            ax.set_xlabel("Number of model params (billions)", fontsize=16)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            if not lang_row.empty:
                g_s_value = lang_row['g_s_score'].values[0]
            # Plot the point
                # X-axis: model size, Y-axis: g_s score
                label = f'{m_name} Instruct' if is_instruct else m_name
                ax.scatter(float(model_size), g_s_value, 
                            color=color, marker=marker, s=200, label=label if lang_index == 0 else "")

            # Format the subplot (only needs to happen once or every time, safe either way)
            if model_label == model_labels[0]: # Formatting on first pass
                ax.set_title(EURO_LLM_LANGS[language].upper(), fontsize=20, fontweight='bold')
                ax.set_ylabel('Stereotype rate', fontsize=16)
                ax.set_xlabel("Params (B)", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(0.9,1.8)
                ax.axhline(1.0, color="red", linestyle="dotted", linewidth=2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks(np.arange(0,15,2))
                ax.set_xlim(left=0)
                
    # 4. Final Layout and Save
    plt.tight_layout()
    
    # Create a global legend (only unique entries)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', 
            bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=20)

    save_path = os.path.join(graphs_folder, f"{subset}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Combined plot saved to: {save_path}")



def plot_inclinations(model_labels, graphs_dir, langs, dfs):
    plt.figure(figsize=(10, 6)) # Adjusted for better proportions
    model_family_store = {}

    graphs_folder = os.path.join(graphs_dir, 'inclinations')
    os.makedirs(graphs_folder, exist_ok=True)

    # load all relevant data from all models 
    for model_label in model_labels:
        df=dfs[model_label]
        df["avg"] = df.mean(axis=1, numeric_only=True)
        
        model_label = model_label.upper()

        m_name, model_size, is_instruct = parse_model_label(model_label)
        # set family index for colour / shape 
        if m_name not in model_family_store:
            # We multiply by 2 so each family jumps ahead by two slots
            model_family_store[m_name] = {"size": [model_size],
                                          "is instruct": [is_instruct],
                                          "df": [df["avg"]]}
        else:
            # Append the new values to the existing lists
            model_family_store[m_name]["size"].append(model_size)
            model_family_store[m_name]["is instruct"].append(is_instruct)
            model_family_store[m_name]["df"].append(df["avg"])
    
    # for each mdoel family, plot the inclination graph with size vairants 
    for m_family in model_family_store.keys():
        family_data = model_family_store[m_family]
        base_cols = {}
        instr_cols = {}
        
        all_labels = [STEREOTYPE_LABELS[i] for i in range(1, 17)]

        for i in range(len(family_data['size'])):
            size = family_data['size'][i]
            is_instr = family_data['is instruct'][i]
            # Ensure the data matches the 1-16 index labels
            data = family_data['df'][i]
            
            if is_instr:
                instr_cols[size] = data
            else:
                base_cols[size] = data

        # 1. Determine how many subplots we need
        has_base = len(base_cols) > 0
        has_instr = len(instr_cols) > 0
        num_plots = int(has_base) + int(has_instr)

        if num_plots == 0:
            continue

        # 2. Dynamic Subplot Creation
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 8), sharey=True, squeeze=False)
        axes = axes.flatten() # Flatten so we can use index [0], [1] regardless of count
        
        curr_ax = 0
        cmap = 'bwr'
        vmax = 0.2
        last_im = None

        # --- Plot Base if exists ---
        if has_base:
            df_base = pd.DataFrame(base_cols).reindex(range(0, 16))
            im = sns.heatmap(df_base, cmap=cmap, center=0.0, vmax=vmax, ax=axes[curr_ax], 
                             square=True, cbar=False, annot=True, fmt=".2f", annot_kws={"fontsize":9})
            axes[curr_ax].set_title(f"{m_family} Base", fontsize=16)
            axes[curr_ax].set_xlabel("# params (billions)", fontsize=14)
            axes[curr_ax].axhline(y=7, color='black', linewidth=2, linestyle='--')
            axes[curr_ax].set_yticklabels(all_labels, rotation=0, fontsize=12)
            last_im = im
            curr_ax += 1

        # --- Plot Instruct if exists ---
        if has_instr:
            df_instr = pd.DataFrame(instr_cols).reindex(range(0, 16))
            im = sns.heatmap(df_instr, cmap=cmap, center=0.0, vmax=vmax, ax=axes[curr_ax], 
                             square=True, cbar=False, annot=True, fmt=".2f", annot_kws={"fontsize":9})
            axes[curr_ax].set_title(f"{m_family} Instruct", fontsize=16)
            axes[curr_ax].set_xlabel("# params (billions)", fontsize=14)
            axes[curr_ax].axhline(y=7, color='black', linewidth=2, linestyle='--')
            
            # If this is the second plot, hide y-ticks to prevent overlap
            if curr_ax > 0:
                axes[curr_ax].tick_params(axis='y', which='both', length=0)
            else:
                axes[curr_ax].set_yticklabels(all_labels, rotation=0, fontsize=12)
            
            last_im = im

        # 3. Add Colorbar relative to the existing axes
        fig.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        fig.colorbar(last_im.get_children()[0], cax=cbar_ax)

        save_path = f"{graphs_folder}/{m_family}_heatmap.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved inclination heatmap: {save_path}")
        plt.close()
    
    
    # print(model_family_store)

def main(model_labels,
         languages,
         results_dir,
         graphs_folder,
         subset,
         sort_by,
         ):
    
    if subset == 'EU':
        selected_langs = list(EU_LANGS.keys())
    else:
        selected_langs = list(languages)
    
    if sort_by == 'family':
        final_langs = [l for l in LANGS_SORTED_BY_FAMILY.keys() if l in selected_langs]
        # Add any selected languages that weren't in the family dict to the end
        extra = [l for l in selected_langs if l not in LANGS_SORTED_BY_FAMILY]
        final_langs.extend(sorted(extra))
    else:
        final_langs = sorted(selected_langs)

    gs_scores = {}
    inclination_scores = {}

    for model_label in model_labels:
        # find results for that model 
        results_folder = os.path.join(results_dir, model_label, "summary_metrics")
        if not os.path.isdir(results_folder):
            print(f"Warning: Folder not found: {results_folder}")
            continue

        def get_safe_path(filename):
            actual_files = {f.lower(): f for f in os.listdir(results_folder)}
            if filename.lower() in actual_files:
                return os.path.join(results_folder, actual_files[filename.lower()])
            return None
        
        try:
            path_rates = get_safe_path('gs_scores.csv')
            path_rankings = get_safe_path('stereotype_rankings.csv')
            path_inclination = get_safe_path('inclinations.csv')

            if all([path_rates, path_rankings, path_inclination]):
                gs_scores[model_label] = pd.read_csv(path_rates)
                stereotype_rankings = pd.read_csv(path_rankings)
                inclination_scores[model_label] = pd.read_csv(path_inclination)
                # Trigger your visualisation functions here
            else:
                missing = [f for f, p in zip(['rates', 'rankings', 'inclination'], 
                           [path_rates, path_rankings, path_inclination]) if not p]
                print(f"Error: Missing files for {model_label}: {missing}")

        except Exception as e:
            print(f"An error occurred while processing {model_label}: {e}")

        plot_stereotype_positions(stereotype_rankings, final_langs, model_label, graphs_folder, sort_by)

    plot_stereotype_rates_per_lang(model_labels, graphs_folder, final_langs, gs_scores, subset)
    plot_inclinations(model_labels, graphs_folder, final_langs, inclination_scores)
                        
if __name__ == "__main__":
    fire.Fire(main)

