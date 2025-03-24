import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import DynamicCache
from .utils import load_data
from .prompts import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_SUFFIX, USER_PROMPT_SUFFIX_WITH_DATA


def save_cache(cache: tuple, cache_dir: str, cache_name: str):
    torch.save(cache, os.path.join(cache_dir, cache_name))


def get_kv_cache_slice(kv_cache, st, ed):
    """
    kv_cache shape: Tuple(layer, 2) +  Tensor(batch_size, num_heads, seq_len, head_dim)
    """
    res = []
    for i in range(len(kv_cache)):
        k = kv_cache[i][0]
        v = kv_cache[i][1]
        k_slice = k[:, :, st:ed, :].detach()
        v_slice = v[:, :, st:ed, :].detach()
        res.append((k_slice, v_slice))
    return res


def merge_kv_caches(cache_list):
    if not cache_list:
        return []

    merged = []
    num_layers = len(cache_list[0])

    for layer_idx in range(num_layers):
        k_list = []
        v_list = []
        for cache in cache_list:
            k, v = cache[layer_idx]
            k_list.append(k)
            v_list.append(v)

        merged_k = torch.cat(k_list, dim=2)
        merged_v = torch.cat(v_list, dim=2)
        merged.append((merged_k, merged_v))

    return merged


def _sys_cache_prep(model, tokenizer):
    system_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n<|im_start|>user\n[[## DATA ##]]\n"
    )
    system_inputs = tokenizer(system_prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    output = model(**system_inputs)
    system_cache = output.past_key_values
    return system_cache


def _cache_prep(
    model,
    tokenizer,
    data_entry: str,
    system_cache: tuple = None,
):

    entry_ids = tokenizer.encode(data_entry, add_special_tokens=False)

    token_ids = torch.tensor(entry_ids, device=model.device).unsqueeze(0)
    past_key_values = DynamicCache.from_legacy_cache(system_cache)

    with torch.inference_mode():
        output = model(token_ids, past_key_values=past_key_values, use_cache=True)

    # remove system cache from past_key_values
    data_cache = get_kv_cache_slice(
        output.past_key_values, system_cache[0][0].shape[2], None
    )
    return data_cache

