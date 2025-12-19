#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:06:26 2025

@author: s2583833
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
    
    if 'Unnamed: 0' in df.columns:
        df = df.set_index('Unnamed: 0')

    available_langs = [l for l in languages if l in df.columns]
    plot_data = df[available_langs]

    plt.figure(figsize=(10, 6)) # Adjusted for better proportions
    
    # center=8 ensures rank 8 is neutral; vmin/vmax keeps the scale consistent
    ax = sns.heatmap(plot_data, 
                     cmap='PuOr_r', 
                     square=True, 
                     center=8, 
                     vmin=1, 
                     vmax=16, 
                     cbar_kws={'label': 'Masculine Rank (1=Most Masc)'})
    
    plt.xlabel('Language', fontsize=14, labelpad=10)
    plt.ylabel('Stereotype', fontsize=14, labelpad=10)

    lang_codes = [EURO_LLM_LANGS[lang].upper() for lang in plot_data.columns]
    
    ax.set_xticks(np.arange(len(lang_codes)) + 0.5)
    ax.set_xticklabels(lang_codes, ha='center', fontsize=10)
    
    # Red line to separate Female stereotypes (top 7) from Male
    plt.axhline(y=7, color='red', linewidth=2, linestyle='--')
    
    # Y-axis labels from the index of the DF
    ax.set_yticks(np.arange(len(plot_data.index)) + 0.5)
    ax.set_yticklabels(plot_data.index, rotation=0, fontsize=10)

    plt.tight_layout()
    
    # Safe filename for OS
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
                g_s_value = lang_row['g_s score'].values[0]
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

        # # for index, model_size in enumerate(sizes):
        # We need to distinguish between Base and Instruct for the two subplots
        base_cols = {}
        instr_cols = {}
        
        # Attribute labels from stereotypes 
        # family_data.index = family_data.index.map(STEREOTYPE_LABELS)
        all_labels = [STEREOTYPE_LABELS[i] for i in range(1,17)]

        for i in range(len(family_data['size'])):
            size = family_data['size'][i]
            is_instr = family_data['is instruct'][i]
            data = family_data['df'][i]
            
            if is_instr:
                instr_cols[size] = data
            else:
                base_cols[size] = data

        # Convert to DataFrames: Rows=Attributes, Cols=Sizes
        df_base = pd.DataFrame(base_cols)
        df_instr = pd.DataFrame(instr_cols)

        # 2. Start Plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
        cmap = 'bwr'
        vmax = 0.2

        # --- Base Models Heatmap ---
        sns.heatmap(df_base, cmap=cmap, center=0.0, vmax=vmax, ax=axes[0], 
                    square=True, cbar=False, annot=True, fmt=".2f", annot_kws={"fontsize":9})
        axes[0].set_title(f"{m_family} Base", fontsize=16)
        axes[0].set_xlabel("# params (billions)", fontsize=14)
        axes[0].axhline(y=7, color='black', linewidth=1)
        
        # --- Instruct Models Heatmap ---
        im1 = sns.heatmap(df_instr, cmap=cmap, center=0.0, vmax=vmax, ax=axes[1], 
                          square=True, cbar=False, annot=True, fmt=".2f", annot_kws={"fontsize":9})
        axes[1].set_title(f"{m_family} Instruct", fontsize=16)
        axes[1].set_xlabel("# params (billions)", fontsize=14)
        axes[1].axhline(y=7, color='black', linewidth=1)

        # 3. Aesthetics & Labels
        # Set Y-labels (attributes) only on the first plot
        axes[0].set_yticklabels(all_labels, rotation=0, fontsize=12)
        axes[1].tick_params(axis='y', which='both', length=0)
        
        # Add Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(im1.collections[0], cax=cbar_ax)  
        cbar.outline.set_visible(False)

        save_path = os.path.join(graphs_folder, f"{m_family}.png")
        plt.savefig(f"{graphs_folder}/{m_family}_heatmap.png", bbox_inches="tight", dpi=300)
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
        results_folder = os.path.join(results_dir, model_label)
        if not os.path.isdir(results_folder):
            print(f"Warning: Folder not found: {results_folder}")
            continue

        def get_safe_path(filename):
            actual_files = {f.lower(): f for f in os.listdir(results_folder)}
            if filename.lower() in actual_files:
                return os.path.join(results_folder, actual_files[filename.lower()])
            return None
        
        try:
            path_rates = get_safe_path('stereotype_rates_per_lang.csv')
            path_rankings = get_safe_path('stereotype_rankings_per_lang.csv')
            path_inclination = get_safe_path('inclinations_per_lang.csv')

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

        # plot_stereotype_positions(stereotype_rankings, final_langs, model_label, graphs_folder, sort_by)

    # plot_stereotype_rates_per_lang(model_labels, graphs_folder, final_langs, gs_scores, subset)
    plot_inclinations(model_labels, graphs_folder, final_langs, inclination_scores)
                        
if __name__ == "__main__":
    fire.Fire(main)

