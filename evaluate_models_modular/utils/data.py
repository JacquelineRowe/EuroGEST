import random

def format_target_stereotype(target_stereotype): 
    # 1. Standardize "None" variants
    if target_stereotype is None or str(target_stereotype).lower() in ["none", "", "[]"]:
        target_stereotype = None
    
    # 2. Convert string representation of list "[4]" to actual list [4]
    elif isinstance(target_stereotype, str) and target_stereotype.startswith("["):
        import ast
        try:
            target_stereotype = ast.literal_eval(target_stereotype)
            target_stereotype = [float(x.strip()) for x in target_stereotype]
        except:
            # Fallback if it's malformed
            target_stereotype = [float(x.strip()) for x in target_stereotype.strip("[]").split(",") if x.strip()]

    # 3. Ensure target_stereotype is ALWAYS a list for .isin()
    if target_stereotype is not None:
        if isinstance(target_stereotype, list):
            target_stereotype = [float(x) for x in target_stereotype]
        else:
            target_stereotype = [float(target_stereotype)]

    return target_stereotype

def get_consistent_indices(dataset, languages, sample_size, target_stereotype, seed:int):
    """
    Identifies a set of indices to evaluate across multiple languages.

    The function first attempts to find a shared set of 'GEST_ID' values that exist 
    across all specified languages (intersection). If a shared set exists, it 
    samples from that intersection to ensure cross-lingual consistency. If no 
    intersection is found, it falls back to independent random sampling for each 
    language.

    Args:
        dataset (datasets.DatasetDict): A Hugging Face DatasetDict where keys 
            are language names and values contain the data.
        eval_languages (list[str]): The list of language keys to process.
        sample_size (int): The number of samples to retrieve. If set to 1, 
            the function retrieves all available common indices.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        dict[str, list[int]]: A dictionary mapping each language to a list 
            of selected indices (either 'GEST_ID' strings or integer row indices).
    """

    random.seed(seed)
    common_indices = None

    target_stereotype = format_target_stereotype(target_stereotype)

    for lang in languages:
        try:
            lang_df = dataset[lang].to_pandas()
            # select only rows with the target stereotype if specified
            if target_stereotype is not None:
                lang_df = lang_df[lang_df['Stereotype_ID'].isin(target_stereotype)]
            current_indices = set(lang_df["GEST_ID"])
            if common_indices is None:
                common_indices = current_indices
            else:
                common_indices = common_indices.intersection(current_indices)
                if common_indices == set():
                    print(f"No common indices found across languages up to {lang}. Falling back to independent sampling.")
                    break
        except KeyError:
            continue

    if common_indices and len(common_indices) > 0:
        common_list = sorted(list(common_indices))
        n = len(common_list) if sample_size == 1 else min(int(sample_size), len(common_list))
        sampled = random.sample(common_list, n)
        return {lang: sampled for lang in languages}

    # Fallback
    fallback_map = {}
    for lang in languages:
        df_len = len(dataset[lang])
        all_ids = list(range(df_len))
        n = sample_size if sample_size == 1 else min(int(sample_size), df_len)
        fallback_map[lang] = random.sample(all_ids, n)
    return fallback_map