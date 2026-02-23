import re
from .setup import SUPPORTED_LANGS
from utils import (
    load_schema_configs)


def define_prompting_options(source_languages, eval_languages, exp):
    
    prompting_options = {}  

    if exp == "none":
        prompting_options["baseline"] = ""
    elif "translation" in exp:
        prompting_languages = source_languages if isinstance(source_languages, list) else [source_languages]
        prompting_options = load_schema_configs(exp)
    else:
        INSTRUCTION_SCHEMAS = load_schema_configs("instruction")
        DEBIASING_SCHEMAS = load_schema_configs("debiasing")
        if exp == "debiasing_multilingual":
            # prompt languages are eval language if it's a generation task, otherwise we use the translation source languages for constructing the prompt 
            prompting_languages = eval_languages
        elif exp == "debiasing_english":
            prompting_languages = ["English"]
        # add instruction part of the prompt 
        for lang in prompting_languages:
            l_code = SUPPORTED_LANGS[lang]
            for instr_key, instruction_prompt in INSTRUCTION_SCHEMAS.items():
                basic_instruction_prompt = INSTRUCTION_SCHEMAS[instr_key][l_code].strip()
                prompting_options[f"{instr_key}_{l_code}"] = re.sub(r"<db>", "", basic_instruction_prompt)
                for db_key, debiasing_prompts in DEBIASING_SCHEMAS['debiasing'].items():
                    debias_prompt_prediction = re.sub(r"<db>", debiasing_prompts[l_code].strip(), basic_instruction_prompt)
                    prompting_options[f"{instr_key}_{db_key}_{l_code}"] = debias_prompt_prediction

    return prompting_options



def build_row_prompts(row, gendered_row, prompting_options, eval_lang, scaffolds, punc_map, eval_task):
    # 1. Prepare Sentence Variations (Gendered vs Neutral)
    eng_sentence = row['Source']
    if gendered_row:
        m_sent, f_sent = row['Masculine'], row['Feminine']
        condition = "G"
    else:
        m_sent, m_sent_noun, f_sent, f_sent_noun = wrap_neutral_sentence(row['Neutral'], eval_lang, scaffolds, punc_map)
        condition = "P"
        if m_sent == f_sent:
            m_sent, f_sent = m_sent_noun, f_sent_noun
            condition = "N"

    model_inputs = {}

    masc_words = m_sent.split()
    fem_words = f_sent.split()
        
    fem_word = None
    masc_word = None
    
    for idx, word in enumerate(fem_words):
        if idx < len(masc_words) and word != masc_words[idx]:
            fem_word = word
            masc_word = masc_words[idx]
            break

    if "translation" in eval_task:
        
        m_sent_masked = re.sub(masculine_word, "______", m_sent)
        f_sent_masked = re.sub(feminine_word, "______", f_sent)

        if m_sent_masked != f_sent_masked:
            print(f"Mismatch found, skipping {m_sent_masked} / {f_sent_masked}")
            return None, None  # Jumps to the start of the next iteration
    
    # 2. Iterate through prompting strategies
        for prompt_id, prompt_data in prompting_options.items():

            id_fem, id_masc, t1, t2 = ("", "", "", "")

            if prompt_id and prompt_id[0].isdigit() and prompt_id.endswith("a_MCQ"):
                mask1 = feminine_word
                mask2 = masculine_word
                id_fem = 1
                id_masc = 2
                t1 = f_sent
                t2 = m_sent
            else:
                mask1 = masculine_word
                mask2 = feminine_word
                id_masc = 1
                id_fem = 2
                t1 = m_sent
                t2 = f_sent

            full_prompt = re.sub(r"<target_lang>", eval_lang, prompt_data)
            full_prompt = re.sub(r"<lang_tag>", SUPPORTED_LANGS[eval_lang], full_prompt)
            full_prompt = re.sub(r"<source>", f'\'{eng_sentence}\'', full_prompt)
            full_prompt = re.sub(r"<target_masked>", f'{m_sent_masked}', full_prompt)
            full_prompt = re.sub(r"<mask1>", f'{mask1}', full_prompt)
            full_prompt = re.sub(r"<mask2>", f'{mask2}', full_prompt)

            full_prompt = re.sub(r"<target1>", f'{t1}', full_prompt)
            full_prompt = re.sub(r"<target2>", f'{t2}', full_prompt)

            full_prompt_m = re.sub(r"<target>", f'{m_sent}.', full_prompt)
            full_prompt_f = re.sub(r"<target>", f'{f_sent}.', full_prompt)

            full_prompt_m = re.sub(r"<mask>", f'{masculine_word}', full_prompt_m)
            full_prompt_f = re.sub(r"<mask>", f'{feminine_word}', full_prompt_f)

            full_prompt_n = ""
            if "3. Both translations are equally correct" in full_prompt_m:
                full_prompt_n = re.sub(r"<option>", f'3', full_prompt_m)
            full_prompt_m = re.sub(r"<option>", f'{id_masc}', full_prompt_m)
            full_prompt_f = re.sub(r"<option>", f'{id_fem}', full_prompt_f)

            if full_prompt_m == full_prompt_f:
                model_input = full_prompt_m
                model_inputs[prompt_id] = model_input
            else:   
                model_input = [full_prompt_m, full_prompt_f]
                model_inputs[prompt_id] = model_input
            if full_prompt_n != "":
                model_inputs[prompt_id].append(full_prompt_n)

            if not hasattr(build_row_prompts, "_already_printed"):
                print(f"\n[ID: {prompt_id}]\n{model_input}\n")
                print("-" * 30)

    elif "debiasing" in eval_task:
        for prompt_id, prompt_data in prompting_options.items():

            id_fem, id_masc, t1, t2 = ("", "", "", "")

            if prompt_id and prompt_id.startswith("selection"):
                if prompt_id.startswith("selection-a"):
                    t1 = f_sent
                    t2 = m_sent
                    id_fem = 1
                    id_masc = 2
                else:
                    t1 = m_sent
                    t2 = f_sent
                    id_fem = 2
                    id_masc = 1
            
            full_prompt = re.sub(r"<target1>", f'{t1}.', prompt_data)
            full_prompt = re.sub(r"<target2>", f'{t2}.', full_prompt)

            full_prompt_m = re.sub(r"<target>", f'{m_sent}.', full_prompt)
            full_prompt_f = re.sub(r"<target>", f'{f_sent}.', full_prompt)

            if prompt_id and prompt_id.startswith("selection"):
                full_prompt_m = re.sub(r"<option>", f'{id_masc}.', full_prompt_m)
                full_prompt_f = re.sub(r"<option>", f'{id_fem}.', full_prompt_f)

            model_input = [full_prompt_m, full_prompt_f]
            model_inputs[prompt_id] = model_input

            if not hasattr(build_row_prompts, "_already_printed"):
                print(f"\n[ID: {prompt_id}]\n{model_input}\n")
                print("-" * 30)

    elif eval_task == "none":
        model_inputs["baseline"] = (m_sent, f_sent)

        if not hasattr(build_row_prompts, "_already_printed"):
            print(f"\nID: baseline\n{model_inputs['baseline']}\n")
            print("-" * 30)

    if not hasattr(build_row_prompts, "_already_printed"):
        print("="*50 + "\n")
        build_row_prompts._already_printed = True

    return model_inputs, condition, masc_word, fem_word



def wrap_neutral_sentence(sentence, eval_lang, SCAFFOLDS, PUNC_MAP):

    m_pron, f_pron = "he said", "she said"
    m_noun, f_noun = "the man said", "the woman said"
    if eval_lang != "English":
        m_pron, f_pron = SCAFFOLDS[m_pron][eval_lang], SCAFFOLDS[f_pron][eval_lang]
        m_noun, f_noun = SCAFFOLDS[m_noun][eval_lang], SCAFFOLDS[f_noun][eval_lang]

    sentence = re.sub(r'^[^\w\s]+|[^\w\s]+\Z', '', sentence.strip())
    p_start, p_end = PUNC_MAP[eval_lang]
    base = f"{p_start}{sentence}{p_end}"

    return (f"{base} {m_pron}", f"{base} {m_noun}", f"{base} {f_pron}", f"{base} {f_noun}")