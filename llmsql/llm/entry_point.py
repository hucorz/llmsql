import re
import json
import traceback
import random
import pandas as pd
import pyarrow as pa
from collections import OrderedDict
import llmsql
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from ..logger import logger


class LLMEntryPoint:
    def __init__(self):
        from .api import APIModel
        from .ktransformers import KTransformers

        self.api_model = APIModel()
        self.llm = KTransformers()

    def query(
        self,
        data: list[dict[str, str]],
        query: str,
        output_format: str,
        use_turbo: bool = True,
        use_cache: bool = True,
        is_full_data: bool = False,
    ):
        logger.info(
            f"LLMEntryPoint query (VECTORIZE:{llmsql.VECTORIZE}, vectorization_stride:{llmsql.VECTORIZATION_STRIDE}) Start"
        )
        try:
            if not llmsql.VECTORIZE:
                output = self.llm.query(data=data, query=query, output_format=output_format)
            else:
                if llmsql.VECTORIZATION_STRIDE == 0:
                    llmsql.VECTORIZATION_STRIDE, partial_output = get_optimal_stride(
                        self.api_model,
                        query,
                        output_format,
                        data,
                    )
                else:
                    partial_output = []

                print(f"optimal_stride: {llmsql.VECTORIZATION_STRIDE}")
                return
                data = data[len(partial_output) :]
                output = partial_output
                output.extend(
                    query_with_stride(
                        self.api_model, data, query, output_format, llmsql.VECTORIZATION_STRIDE
                    )
                )
            output = [json.dumps(item) for item in output]
        except Exception as e:
            traceback.print_exc()
            raise e
        logger.info(
            f"LLMEntryPoint query (VECTORIZE:{llmsql.VECTORIZE}, vectorization_stride:{llmsql.VECTORIZATION_STRIDE}) Complete"
        )
        return output

    def chat(self, messages: list[dict[str, str]]):
        if not llmsql.VECTORIZE:
            response = self.llm.chat(messages)
        else:
            response = self.api_model.chat(messages)
        logger.info(f"LLMEntryPoint chat\nResponse:{response}\n")
        return response


def get_optimal_stride(model, query: str, output_format: str, data: list[dict[str, str]]):
    data_length = len(data)
    # if data_length < 2048:  # actually, duckdb's max rows is 2048 when using arrow mode udf
    #     return 1, []
    # sample_data = data[: data_length // 8]
    sample_data = data[:256]

    strides = [1, 2, 4, 8]

    results = {}
    accuracies = {}

    logger.info(f"Starting optimal stride determination with {len(sample_data)} sample data points")

    # stride=1 baseline
    print(f"Testing with stride=1 (baseline)...")
    baseline_results = query_with_stride(model, sample_data, query, output_format, 1)
    results[1] = baseline_results
    accuracies[1] = 100.0

    # test other stride
    for stride in strides[1:]:
        print(f"Testing with stride={stride}...")
        stride_results = query_with_stride(model, sample_data, query, output_format, stride)
        results[stride] = stride_results

        accuracy = calculate_accuracy(baseline_results, stride_results)
        accuracies[stride] = accuracy

        logger.info(f"Stride={stride} accuracy: {accuracy:.2f}%")

        print(f"Stride={stride} accuracy: {accuracy:.2f}%")

    # find optimal stride
    optimal_stride = 1
    for stride in sorted(strides, reverse=True):
        if accuracies[stride] >= 95.0:
            optimal_stride = stride
            break

    logger.info(
        f"All stride accuracies: {json.dumps({s: round(accuracies[s], 2) for s in accuracies})}"
    )
    logger.info(
        f"Optimal stride selected: {optimal_stride} (accuracy: {accuracies[optimal_stride]:.2f}%)"
    )

    print(f"Optimal stride: {optimal_stride} (accuracy: {accuracies[optimal_stride]:.2f}%)")
    return optimal_stride, results[optimal_stride]


def process_batch(model, batch, query, output_format):
    for idx in range(len(batch)):
        batch[idx] = OrderedDict([("id", idx + 1)] + list(batch[idx].items()))  # id must be first

    data_entry = "\n".join([json.dumps(item, ensure_ascii=False) for item in batch])

    user_prompt = USER_PROMPT.format(
        data_entry=data_entry, query=query, output_format=output_format
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # print("===" * 10)
    # print(SYSTEM_PROMPT)
    # print("===" * 10)
    # print(user_prompt)

    response = model.chat(messages)
    return parse_response(response, len(batch))


def query_with_stride(model, data, query, output_format, stride, max_workers=10):
    results = []
    batches = []

    for i in range(0, len(data), stride):
        batch = data[i : min(i + stride, len(data))]
        batches.append(batch)

    total_batches = len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(
            tqdm(
                executor.map(
                    lambda batch: process_batch(model, batch, query, output_format),
                    batches,
                ),
                total=total_batches,
                desc=f"Processing batches (stride={stride})",
                unit="batch",
            )
        )

    for batch_result in batch_results:
        results.extend(batch_result)

    return results


def parse_response(response, expected_count):
    results = [{} for _ in range(expected_count)]

    data_pattern = r"==== DATA(\d+)_RESULT ====\s*(.*?)(?=====|$)"
    data_matches = re.findall(data_pattern, response, re.DOTALL)

    for data_id_str, data_content in data_matches:
        try:
            data_id = int(data_id_str)
            if 1 <= data_id <= expected_count:
                field_pattern = r"\[\[## (.*?) ##\]\]\s*(.*?)(?=\[\[##|$)"
                field_matches = re.findall(field_pattern, data_content, re.DOTALL)

                fields = {}
                for field_name, field_value in field_matches:
                    fields[field_name.strip()] = field_value.strip()

                results[data_id - 1] = fields
        except ValueError:
            continue
    return results


def calculate_accuracy(baseline_results, test_results):
    """
    计算 test_results 与 baseline_results 相比的准确率。

    Args:
        baseline_results: 基准结果 (stride=1)
        test_results: 要与基准进行比较的结果

    Returns:
        准确率百分比 (0-100)
    """
    if len(baseline_results) != len(test_results):
        # 如果长度不匹配，调整较短的一个
        min_len = min(len(baseline_results), len(test_results))
        baseline_results = baseline_results[:min_len]
        test_results = test_results[:min_len]

    correct = 0
    total = 0

    for baseline, test in zip(baseline_results, test_results):
        # 比较结果中的每个字段
        for field in baseline:
            if field in test and baseline[field] == test[field]:
                correct += 1
            total += 1

    if total == 0:
        return 0.0

    return (correct / total) * 100.0
