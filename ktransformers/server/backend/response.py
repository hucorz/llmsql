import os
import json
import time
import torch
from transformers import DynamicCache
from .utils.utils import parse_output_format_fields, data_dumps
from .utils.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_SUFFIX,
    USER_PROMPT_SUFFIX_WITH_DATA,
)
from .utils.cache import merge_kv_caches
from .trie import build_trie_from_format
from ktransformers.server.config.log import logger


def data_entries_dumps(data_entries: list[dict]):
    return "\n".join(
        [json.dumps(data, ensure_ascii=False, sort_keys=True) for data in data_entries]
    )


def extend_input_ids(
    interface,
    tokenizer,
    generate_token_buffer,
    finish_fields_cnt,
    output_format,
    vectorization_stride: int = 1,
):
    format_keys = list(output_format.keys())
    fields_cnt = len(format_keys)
    cur_field_idx = int(finish_fields_cnt % fields_cnt)
    cur_field = format_keys[cur_field_idx]
    cur_data_idx = int(finish_fields_cnt / fields_cnt)
    # print(f"cur_field_idx: {cur_field_idx}")
    # print(f"cur_data_idx: {cur_data_idx}")

    generate_str_buffer = tokenizer.decode(generate_token_buffer)
    extend_str = ""

    if output_format[cur_field]["type"] != "Literal":  # not Literal type
        if "\n" not in generate_str_buffer:
            return False, None, None  # continue decode
    else:  # Literal type or bool type
        target_category = output_format[cur_field]["trie"].search_unique_category(
            generate_str_buffer
        )
        if not target_category:
            return False, None, None  # continue decode

        extend_str += target_category[len(generate_str_buffer) :] + "\n"
        interface.profiler.inc("trie_hit")

    format_content = ""
    if cur_field_idx == fields_cnt - 1 and cur_data_idx < vectorization_stride - 1:
        format_content = f"==== DATA{cur_data_idx + 2}_RESULT ====\n[[## {format_keys[0]} ##]]\n"
    elif cur_field_idx < fields_cnt - 1:
        format_content = f"[[## {format_keys[cur_field_idx + 1]} ##]]\n"
    else:
        format_content = f"==== ALL_COMPLETE ===="

    extend_str += format_content
    extend_ids = tokenizer.encode(extend_str, add_special_tokens=False)

    return True, extend_ids, generate_str_buffer + extend_str


def _generate(
    interface,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    kv_cache: tuple,
    output_format: dict,
    vectorization_stride: int = 1,
    max_new_tokens: int = 200,
    use_turbo: bool = True,
):
    response: list[str] = []
    generate_token_buffer = []

    input_ids = input_ids.to(model.device)
    past_key_values = DynamicCache.from_legacy_cache(kv_cache)

    with torch.inference_mode():
        for idx in range(max_new_tokens):
            if idx == 0:
                interface.profiler.start_timer("prefill")
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1))
            interface.profiler.inc("decode")
            if next_token_id == tokenizer.eos_token_id:
                break
            if use_turbo:
                generate_token_buffer.append(next_token_id)
                field_finish, extend_ids, generate_str = extend_input_ids(
                    interface,
                    tokenizer,
                    generate_token_buffer,
                    len(response),
                    output_format,
                    vectorization_stride,
                )
                if not field_finish:
                    input_ids = torch.tensor([next_token_id], device=model.device).unsqueeze(0)
                else:
                    input_ids = torch.tensor(
                        [next_token_id] + extend_ids, device=model.device
                    ).unsqueeze(0)
                    generate_token_buffer = []
                    response.append(generate_str)
            else:
                input_ids = torch.tensor([next_token_id], device=model.device).unsqueeze(0)
                response.append(tokenizer.decode(next_token_id))
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
            if idx == 0:
                interface.profiler.pause_timer("prefill")
    return "".join(response)


def generate_turbo(
    interface,
    model,
    tokenizer,
    input_str: str,
    output_format: str,
    kv_cache: tuple = None,
    vectorization_stride: int = 1,
    max_new_tokens: int = 200,
    add_generation_prompt: bool = False,
):
    """
    - case1: without cache
        input_str need to contain both system prompt and user prompt.
    - case2: with cache (only system prompt cache, exclude data info)
        the input_str need to contain the data info, query and format.
    - case3: with cache (both system prompt cache and data info)
        the input_str need to contain query and format.
    """
    output_format = parse_output_format_fields(output_format)

    for k, v in output_format.items():
        if v["type"] == "Literal":
            output_format[k]["trie"] = build_trie_from_format(v["values"])
        elif v["type"] == "bool":
            output_format[k]["trie"] = build_trie_from_format("Literal[true, false]")
        else:
            output_format[k]["trie"] = None

    if add_generation_prompt:
        input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    pre_extend_text = f"==== DATA1_START ====\n[[## {list(output_format.keys())[0]} ##]]\n"
    input_str += pre_extend_text
    inputs = tokenizer(input_str, return_tensors="pt").to(model.device)
    return pre_extend_text + _generate(
        interface,
        model,
        tokenizer,
        inputs["input_ids"],
        kv_cache,
        output_format,
        vectorization_stride,
        max_new_tokens=max_new_tokens,
        use_turbo=True,
    )


def response_normal(
    interface,
    model,
    tokenizer,
    data: list[dict],
    query: str,
    output_format: str,
    max_new_tokens: int = 200,
):
    data_entry = data_entries_dumps(data)
    user_prompt = USER_PROMPT.format(
        data_entry=data_entry, query=query, output_format=output_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    return _generate(
        interface,
        model,
        tokenizer,
        inputs["input_ids"],
        None,
        output_format,
        max_new_tokens=max_new_tokens,
        use_turbo=False,
    )


def response_normal_with_system_cache(
    interface,
    model,
    tokenizer,
    data: list[dict],
    query: str,
    output_format: str,
    system_cache: tuple,
    max_new_tokens: int = 200,
):
    """
    cache content may like:
    <|im_start|>system
    ....
    <|im_end|>
    <|im_start|>user
    [[## DATA ##]]
    """
    assert system_cache, "system_cache must be provided"

    data_entry = data_entries_dumps(data)

    input_str = USER_PROMPT_SUFFIX_WITH_DATA.format(
        data_entry=data_entry, query=query, output_format=output_format
    )
    input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_str, return_tensors="pt").to(model.device)
    return _generate(
        interface,
        model,
        tokenizer,
        inputs["input_ids"],
        system_cache,
        output_format,
        max_new_tokens=max_new_tokens,
        use_turbo=False,
    )


def response_turbo_without_cache(
    interface,
    model,
    tokenizer,
    data: list[dict],
    query: str,
    output_format: str,
    vectorization_stride: int = 1,
    max_new_tokens: int = 200,
):
    data_entry = data_entries_dumps(data)

    user_prompt = USER_PROMPT.format(
        data_entry=data_entry, query=query, output_format=output_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        output_format,
        None,
        vectorization_stride,
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_system_cache(
    interface,
    model,
    tokenizer,
    data: list[dict],
    query: str,
    output_format: str,
    system_cache: tuple,
    vectorization_stride: int = 1,
    max_new_tokens: int = 200,
):
    """
    cache content may like:
    <|im_start|>system
    ....
    <|im_end|>
    <|im_start|>user
    [[## DATA ##]]
    """
    assert system_cache, "system_cache must be provided"

    data_entry = data_entries_dumps(data)

    input_str = USER_PROMPT_SUFFIX_WITH_DATA.format(
        data_entry=data_entry, query=query, output_format=output_format
    )
    input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        output_format,
        system_cache,
        vectorization_stride,
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_all_cache(
    interface,
    model,
    tokenizer,
    query: str,
    output_format: str,
    system_cache: tuple,
    data_cache: tuple,
    max_new_tokens: int = 200,
):
    """
    kv_cache content may like:
    <|im_start|>system
    ....
    <|im_end|>
    <|im_start|>user
    [[## DATA ##]]
    ...
    ...
    """
    assert system_cache and data_cache, "system_cache and data_cache must be provided"

    user_prompt = USER_PROMPT_SUFFIX.format(query=query, output_format=output_format)

    input_str = "\n" + user_prompt + f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        output_format,
        merge_kv_caches([system_cache, data_cache]),
        max_new_tokens=max_new_tokens,
    )
