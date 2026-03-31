import ast
import os
from pathlib import Path
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from matplotlib.lines import Line2D

DEFAULT_RESULTS_FOLDER = Path.home() / "Desktop" / "translation_initial_results"
results_folder = Path(os.getenv("EUROGEST_RESULTS_FOLDER", DEFAULT_RESULTS_FOLDER))

os.makedirs('results_outputs', exist_ok=True)

EURO_LLM_LANGS = {
    'Bulgarian': 'bg', 'Catalan': 'ca', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 
    'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 
    'Galician': 'gl', 'German': 'de', 'Greek': 'el', 'Hungarian': 'hu', 'Irish': 'ga', 
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt', 'Norwegian': 'no', 
    'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 
    'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk'
}

results_dfs = {}

def ensure_list(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, list):
        return value
    return []

def sum_log_probs(sequence):
    if not sequence:
        return 0.0
    return float(np.sum(sequence))

def logistic_masc_rate(masc_log_prob, fem_log_prob):
    delta = fem_log_prob - masc_log_prob
    delta = np.clip(delta, -60, 60)
    return 1.0 / (1.0 + np.exp(delta))


def safe_split(value):
    if value is None:
        return []
    text = str(value).strip()
    return text.split() if text else []


def find_first_divergence(masc_tokens, fem_tokens):
    min_len = min(len(masc_tokens), len(fem_tokens))
    for idx in range(min_len):
        if masc_tokens[idx] != fem_tokens[idx]:
            return idx
    return max(0, min_len - 1)


def select_token(tokens, index):
    if not tokens:
        return ""
    if 0 <= index < len(tokens):
        return str(tokens[index]).strip()
    return str(tokens[-1]).strip()


def extract_target_words(original_masc, original_fem, prompt_id):
    masc_words = safe_split(original_masc)
    fem_words = safe_split(original_fem)
    divergence_idx = find_first_divergence(masc_words, fem_words)
    masculine_word = select_token(masc_words, divergence_idx)
    feminine_word = select_token(fem_words, divergence_idx)
    neutral_word = "3" if "6" in prompt_id else None
    return masculine_word, feminine_word, neutral_word


def classify_generation_output(generated_text, masculine_word, feminine_word, neutral_word):
    normalized_text = str(generated_text).lower()
    if masculine_word and masculine_word.lower() in normalized_text:
        return "m"
    if feminine_word and feminine_word.lower() in normalized_text:
        return "f"
    if neutral_word and neutral_word.lower() in normalized_text:
        return "n"
    return "invalid"

def init_context_counts():
    return {"m": 0, "f": 0, "n": 0, "invalid": 0}

def plot_token_surprisal(df, language, num_text_samples=5):
    print(len(df))
    plot_df = df.head(num_text_samples)
    print(len(plot_df))

    fig, axes = plt.subplots(num_text_samples, 1, figsize=(10, 10), sharex=True)
    
    # Ensure axes is iterable even if n_samples is 1
    if num_text_samples == 1:
        axes = [axes]

    # Select a subset to label to avoid total clutter
    sample_indices = df.index[:num_text_samples] 
    colors = cm.get_cmap('tab10', len(df))
    
    pre_window, post_window = 5, 4

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if "language" in df.columns:
            language = df['language'].iloc[i]

        ax = axes[i]
        current_color = colors(i % 10) # Cycle through tab10 colors
        
        m_toks = ensure_list(row.get('masc_tokens', []))
        f_toks = ensure_list(row.get('fem_tokens', []))
        m_probs = ensure_list(row.get('masc_log_probs_sequence', []))
        f_probs = ensure_list(row.get('fem_log_probs_sequence', []))

        diverge_idx = find_first_divergence(m_toks, f_toks)
        window_start = max(0, diverge_idx - pre_window)
        window_end = diverge_idx + post_window + 1

        m_probs_sub = m_probs[window_start:window_end]
        f_probs_sub = f_probs[window_start:window_end]
        m_toks_sub = m_toks[window_start:window_end]
        f_toks_sub = f_toks[window_start:window_end]

        div_local_idx = max(0, min(diverge_idx - window_start, len(m_probs_sub) - 1)) if m_probs_sub else 0
        x_range = np.arange(len(m_probs_sub))

        # --- PLOTTING ---
        # Masculine: Solid Line
        ax.plot(x_range, m_probs_sub, marker='o', color=current_color, 
                linestyle='-', alpha=0.5, linewidth=1.5, label="Masculine")
        
        # Feminine: Dashed Line
        ax.plot(x_range, f_probs_sub, marker='x', color=current_color, 
                linestyle='--', alpha=0.5, linewidth=1.5, label="Feminine")

        # --- ANNOTATION LOGIC (Divergence Point) ---
        # --- FULL SEQUENCE ANNOTATION ---
        if idx in sample_indices:
            for j in range(len(m_probs_sub)):
                # Relative index of divergence for bolding
                is_divergence = (j+1 == pre_window)
                weight = 'bold' if is_divergence else 'normal'
                
                # Plot Masculine Token (Above)
                if j < len(m_toks_sub)-1:
                    ax.annotate(
                        m_toks_sub[j+1],
                        xy=(j, m_probs_sub[j]),
                        xytext=(0, 12),
                        textcoords='offset points',
                        color=current_color,
                        fontweight=weight,
                        ha='center',
                        fontsize=8,
                        rotation=30 # Rotated for better horizontal packing
                    )
                
                # Plot Feminine Token (Below)
                if j < len(f_toks_sub)-1:
                    ax.annotate(
                        f_toks_sub[j+1],
                        xy=(j, f_probs_sub[j]),
                        xytext=(0, -20),
                        textcoords='offset points',
                        color=current_color,
                        fontweight=weight,
                        ha='center',
                        fontsize=8,
                        rotation=-30
                    )

            # Formatting
            ax.set_ylim(-10, 1)
            ax.set_title(f"Token Surprisal Example from {language}", pad=10)
            ax.set_ylabel("Log Probability")
            
            # Corrected Ticks: t+0 is at the divergence index
            ax.set_xticks(range(pre_window + post_window + 1))
            xtick_labels = [f"t{j-(pre_window-1)}" for j in range(pre_window + post_window + 1)]
            ax.set_xticklabels(xtick_labels)
    
            # Add a vertical line to highlight the divergence point
            ax.axvline(x=pre_window-1, color='black', linestyle=':', alpha=0.3)

            ax.grid(True, alpha=0.15)

    plt.suptitle(f"Token Surprisal Comparison", fontsize=16, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"results_outputs/token_surprisal_examples.png", bbox_inches='tight', dpi=300)
    plt.show()



for language in EURO_LLM_LANGS.keys():
    file_path = results_folder / f"{language.lower()}.csv"
    try:
        results_df = pd.read_csv(file_path)
    except FileNotFoundError:
        continue

    for column in [
        "masc_tokens",
        "fem_tokens",
        "masc_log_probs_sequence",
        "fem_log_probs_sequence",
    ]:
        results_df[column] = results_df[column].apply(ensure_list)

    results_dfs[language] = results_df

prompt_ids_order = ['baseline','1','2','2a','3','4_MCQ','5_MCQ','6_MCQ','4a_MCQ','5a_MCQ','6a_MCQ']

results_generated_summaries = {}
results_logs_summaries = {}

for lang, results_df in results_dfs.items():
    dataset_obj = load_dataset("utter-project/EuroGEST", split=lang)
    original_data_df = dataset_obj.to_pandas()
    id_to_data = original_data_df.set_index('GEST_ID').to_dict('index')

    main_df = results_df.copy()
    main_df['masc_total_log_prob'] = main_df['masc_log_probs_sequence'].apply(sum_log_probs)
    main_df['fem_total_log_prob'] = main_df['fem_log_probs_sequence'].apply(sum_log_probs)
    main_df['masc_rate'] = main_df.apply(
        lambda row: logistic_masc_rate(row['masc_total_log_prob'], row['fem_total_log_prob']), axis=1
    )
    results_summary = []
    results_summary_generated = {}

    for prompt_id in prompt_ids_order:
        prompt_df = main_df[main_df['Prompt ID'] == prompt_id]
        
        # Note: Changed st_id range if your dicts are 1-17
        stereotypes_logprobs = {i: 0.0 for i in range(1, 17)}
        fem_counts = init_context_counts()
        masc_counts = init_context_counts()

        for st_id in range(1, 17):
            subset = prompt_df[prompt_df['Stereotype_ID'] == st_id]
            avg_masc_rate = subset['masc_rate'].mean()
            if not np.isnan(avg_masc_rate):
                stereotypes_logprobs[st_id] = avg_masc_rate

            for _, row in subset.iterrows():
                original_row = id_to_data.get(row['GEST_ID'])
                if not original_row:
                    continue

                masculine_word, feminine_word, neutral_word = extract_target_words(
                    original_row.get('Masculine'), original_row.get('Feminine'), prompt_id
                )

                label = classify_generation_output(
                    row.get('generated_text', ""), masculine_word, feminine_word, neutral_word
                )

                if st_id <= 7:
                    fem_counts[label] += 1
                else:
                    masc_counts[label] += 1

        # Store generated counts (ensure prompt_id is a string for JSON keys)
        results_summary_generated[str(prompt_id)] = {
            'fem_context_counts': fem_counts,
            'masc_context_counts': masc_counts
        }
        
        # Calculate averages for JSON summary
        fem_masc_rates = [stereotypes_logprobs[i] for i in range(1, 8)]
        masc_masc_rates = [stereotypes_logprobs[i] for i in range(8, 17)] # Adjusted to catch all up to 16

        results_summary.append({
            'prompt_id': str(prompt_id),
            'fem_context_bias': float(sum(fem_masc_rates) / len(fem_masc_rates)),
            'masc_context_bias': float(sum(masc_masc_rates) / len(masc_masc_rates))
        })

    # --- JSON Writing Section ---
    language_payload = {
        'language': lang,
        'logs_summaries': results_summary,
        'generated_summaries': results_summary_generated
    }

    file_path = f'results_outputs/{lang}_results.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(language_payload, f, indent=4)
    
    print(f"Saved results for {lang} to {file_path}")


# plot surprisals too to look
# sample one sentence randomly from each langugae from the baseline category
baseline_df_samples = pd.DataFrame(columns = results_df.columns.tolist() + ["language"])

for lang, lang_df in results_dfs.items():
    sample_df = lang_df[lang_df['Prompt ID'] == 'baseline'].sample(1, random_state=42)
    sample_df['language'] = lang
    baseline_df_samples = pd.concat([baseline_df_samples, sample_df], ignore_index=True)

baseline_df_samples['avg_masc_log_prob'] = baseline_df_samples['masc_log_probs_sequence'].apply(
    lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
)


plot_token_surprisal(baseline_df_samples, lang, 5)

