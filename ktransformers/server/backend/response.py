import os
import json
import torch
from transformers import DynamicCache
from .utils.utils import parse_user_format_fields, data_dumps
from .utils.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_SUFFIX,
    USER_PROMPT_SUFFIX_WITH_DATA,
)
from .utils.cache import merge_kv_caches
from .trie import build_trie_from_format
from ktransformers.server.config.log import logger


def is_bool_token(token, tokenizer):
    text = tokenizer.decode(token)
    return text.strip().lower() in ["true", "false"], text


def extend_input_ids(
    interface, tokenizer, cur_generate_tokens, finish_fields_cnt, output_format, device
):
    format_keys = list(output_format.keys())
    fields_cnt = len(format_keys)
    cur_field_idx = int(finish_fields_cnt % fields_cnt)
    cur_field = format_keys[cur_field_idx]
    last_token_id = cur_generate_tokens[-1]

    cur_generate_str = tokenizer.decode(cur_generate_tokens)
    extend_str = ""

    if output_format[cur_field]["type"] != "Literal":
        # not Literal type
        if "\n" not in cur_generate_str:
            return False, None, None  # continue decode
    else:
        # Literal type or bool type
        target_category = output_format[cur_field]["trie"].search_unique_category(cur_generate_str)
        if not target_category:
            return False, None, None  # continue decode

        extend_str += target_category[len(cur_generate_str) :] + "\n"
        interface.profiler.inc("trie_hit")

    format_content = ""
    if cur_field_idx < fields_cnt - 1:
        format_content = f"[[## {format_keys[cur_field_idx + 1]} ##]]\n"
    else:
        format_content = f"[[## COMPLETE ##]]"

    extend_str += format_content
    extend_ids = tokenizer.encode(extend_str, add_special_tokens=False)
    input_ids = torch.tensor([last_token_id] + extend_ids, device=device).unsqueeze(0)

    return True, input_ids, cur_generate_str + extend_str


def _generate(
    interface,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    kv_cache: tuple,
    output_format: dict,
    max_new_tokens: int = 200,
    use_turbo: bool = True,
):
    response = []
    cur_generate_token = []

    input_ids = input_ids.to(model.device)
    past_key_values = DynamicCache.from_legacy_cache(kv_cache)

    with torch.inference_mode():
        for idx in range(max_new_tokens):
            if idx == 0:
                interface.profiler.start_timer("prefill")
            # logger.info(f"\ncur response:\n{''.join(response)}")
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1))
            if next_token_id == tokenizer.eos_token_id:
                break
            if use_turbo:
                cur_generate_token.append(next_token_id)
                field_finish, input_ids, generate_str = extend_input_ids(
                    interface,
                    tokenizer,
                    cur_generate_token,
                    len(response),
                    output_format,
                    model.device,
                )
                if not field_finish:
                    input_ids = torch.tensor([next_token_id], device=model.device).unsqueeze(0)
                else:
                    cur_generate_token = []
                    response.append(generate_str)
            else:
                input_ids = torch.tensor([next_token_id], device=model.device).unsqueeze(0)
                # if "\n" in tokenizer.decode(next_token_id) or "\t" in tokenizer.decode(next_token_id):
                #     print(repr(tokenizer.decode(next_token_id)))
                response.append(tokenizer.decode(next_token_id))
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
            interface.profiler.inc("decode")
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
    max_new_tokens: int = 200,
    add_generation_prompt: bool = False,
):
    """
    - case1: without cache
        input_str need to contain both system prompt and user prompt
    - case2: with cache (only system prompt cache, exclude data info)
        the input_str need to contain the data info, like:
        ==================
        [[## DATA ##]]
        {data_entry}
        [[## QUERY ##]]
        {query}
        [[## FORMAT ##]]
        {output_format}
        ==================
    - case3: with cache (both system prompt cache and data info)
        the input_str is like:
        ==================
        [[## QUERY ##]]
        {query}
        [[## FORMAT ##]]
        {output_format}
        ==================

    """
    output_format = parse_user_format_fields(output_format)
    # print(output_format)

    for k, v in output_format.items():
        if v["type"] == "Literal":
            output_format[k]["trie"] = build_trie_from_format(v["values"])
        elif v["type"] == "bool":
            output_format[k]["trie"] = build_trie_from_format("Literal[true, false]")
        else:
            output_format[k]["trie"] = None

    if add_generation_prompt:
        input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    pre_extend_text = f"[[## {list(output_format.keys())[0]} ##]]\n"
    input_str += pre_extend_text
    # print(f"input_str:\n{input_str}")
    inputs = tokenizer(input_str, return_tensors="pt").to(model.device)
    return pre_extend_text + _generate(
        interface,
        model,
        tokenizer,
        inputs["input_ids"],
        kv_cache,
        output_format,
        max_new_tokens=max_new_tokens,
        use_turbo=True,
    )


def response_normal(
    interface,
    model,
    tokenizer,
    data: dict,
    query: str,
    output_format: str,
    max_new_tokens: int = 200,
):
    data_entry = json.dumps(data, ensure_ascii=False, sort_keys=True)
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

    # use model.generate, which is unable to monitor the decoding times
    # with torch.inference_mode():
    #     outputs = model.generate(
    #         **inputs,
    #         do_sample=False,
    #         max_new_tokens=max_new_tokens,
    #         top_p=None,
    #         top_k=None,
    #         temperature=None,
    #     )
    # input_length = inputs["input_ids"].shape[-1]
    # generated_tokens = outputs[:, input_length:]
    # return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def response_turbo_without_cache(
    interface,
    model,
    tokenizer,
    data: dict,
    query: str,
    output_format: str,
    max_new_tokens: int = 200,
):
    data_entry = json.dumps(data, ensure_ascii=False, sort_keys=True)

    user_prompt = USER_PROMPT.format(
        data_entry=data_entry, query=query, output_format=output_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # logger.info(f"Input str:\n{input_str}\n")
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        output_format,
        None,
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_system_cache(
    interface,
    model,
    tokenizer,
    data: dict,
    query: str,
    user_format: str,
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
    data_entry = json.dumps(data, ensure_ascii=False, sort_keys=True)

    input_str = USER_PROMPT_SUFFIX_WITH_DATA.format(
        data_entry=data_entry, query=query, output_format=user_format
    )
    input_str += f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        user_format,
        system_cache,
        max_new_tokens=max_new_tokens,
    )


def response_turbo_with_all_cache(
    interface,
    model,
    tokenizer,
    query: str,
    user_format: str,
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

    user_prompt = USER_PROMPT_SUFFIX.format(query=query, output_format=user_format)

    input_str = "\n" + user_prompt + f"<|im_end|>\n<|im_start|>assistant\n"
    return generate_turbo(
        interface,
        model,
        tokenizer,
        input_str,
        user_format,
        merge_kv_caches([system_cache, data_cache]),
        max_new_tokens=max_new_tokens,
    )
