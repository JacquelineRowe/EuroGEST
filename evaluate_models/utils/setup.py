import os
import random
import torch
import numpy as np
import json

def setup_environment(seed:int):
    # Redirect cache
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
    
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def _get_config_path(filename: str):
    """Helper to resolve paths relative to the project root."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'configs', filename)

def load_lang_config():
    """Loads language mappings from the configs directory."""
    with open(_get_config_path('supported_languages.json'), 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_scaffolds_configs():
    """
    Centralized loader for experiment configurations.
    Returns:
        tuple: (punc_map, scaffolds, schemas)
    """
    with open(_get_config_path('punc_map.json'), 'r', encoding='utf-8') as f:
        punc_map = json.load(f)
    with open(_get_config_path('prompt_scaffolds.json'), 'r', encoding='utf-8') as f:
        scaffolds = json.load(f)
        
    return punc_map, scaffolds

def load_schema_configs(extension):
    with open(_get_config_path(f'prompting_schemas_{extension}.json'), 'r', encoding='utf-8') as f:
        schemas = json.load(f)

    return schemas

SUPPORTED_LANGS = load_lang_config()
