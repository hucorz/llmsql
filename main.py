#!/usr/bin/env python
import argparse
import json

import numpy as np
import pandas as pd

import llmsql
from llmsql import LLM


def load_data(file_path):
    """加载数据集并提取需要的列"""
    data = json.load(open(file_path, "r"))
    return [d["abstract"] for d in data]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM SQL Demo with customizable parameters"
    )

    # 必需参数
    parser.add_argument(
        "--data-file",
        type=str,
        # default="/fs/fast/share/pingtai_cc/prompt-cache-test/arxiv-metadata-oai-snapshot-sample.json",
        default="/fs/fast/share/pingtai_cc/llmsql/LLM-SQL-Demo/datasets/ratebeer_reviews.csv",
        help="Path to the CSV data file",
    )

    # LLM配置参数
    parser.add_argument(
        "--model",
        type=str,
        # default="openai/Qwen/Qwen2.5-7B-Instruct",
        default="openai/deepseek-ai/DeepSeek-V3",
        help="LLM model name",
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.siliconflow.cn/v1/",
        help="API base URL",
    )

    parser.add_argument(
        "--api-key", type=str, required=True, help="API key for authentication"
    )

    parser.add_argument(
        "--sample-ratio", type=float, default=0.2, help="Sample ratio for the dataset"
    )

    parser.add_argument(
        "--threshold", type=float, default=0.8, help="CV threshold for clustering"
    )

    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")

    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of samples to process"
    )

    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials for statistics"
    )

    parser.add_argument("--n-adjust", type=int, default=1, help="Number of adjustments")

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化LLM
    print("Initializing LLM...")
    llmsql.init(
        llm=LLM(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
        )
    )

    # 读取数据
    print(f"Reading data from {args.data_file}")
    # data = load_data(args.data_file)
    # texts = pd.Series(data)
    df = pd.read_csv(args.data_file)
    texts = df["review/text"][0 : args.n_samples]

    # 创建索引
    print("Creating index...")
    texts.llm.create_index(
        sample_ratio=args.sample_ratio,
        threshold=args.threshold,
        n_clusters=args.n_clusters,
    )

    # 保存初始聚类
    initial_clusters = texts.llm._estimator.text_index.clusters.copy()

    # 定义查询
    # query = "Tell me whether the abstract of this paper is {about_math: bool}"
    query = "Tell me whether the review is {positive: bool}"

    print("\nRunning experiments...")

    # 1. 不调整抽取 n_trials 次
    print("\n1. 不调整抽样...")
    results_no_adjust = []
    for i in range(args.n_trials):
        results_no_adjust.append(texts.llm.sum(query, approx=True, adjust=False))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n_trials} trials completed")

    # 2. 调整一次
    print(f"\n2. 执行{args.n_adjust}次调整...")
    for _ in range(args.n_adjust):
        texts.llm.sum(query, approx=True, adjust=True)

    # 保存调整后的聚类
    adjusted_clusters = texts.llm._estimator.text_index.clusters.copy()

    # 3. 在调整后不调整抽取 n_trials 次
    print("\n3. 调整后不调整抽样...")
    results_after_adjust = []
    for i in range(args.n_trials):
        results_after_adjust.append(texts.llm.sum(query, approx=True, adjust=False))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n_trials} trials completed")

    # 打印统计结果
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)

    # 计算并打印统计数据
    stats = {}
    for name, results in [
        ("Before Adjustment", results_no_adjust),
        ("After Adjustment", results_after_adjust),
    ]:
        mean = np.mean(results)
        std = np.std(results)
        stats[name] = {"mean": mean, "std": std}

        print(f"\n{name}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")

    # 计算改进
    improvement = (
        abs(stats["After Adjustment"]["std"] - stats["Before Adjustment"]["std"])
        / stats["Before Adjustment"]["std"]
        * 100
    )
    print(f"\nStd Improvement after adjustment: {improvement:.2f}%")

    # 获取所有样本的真实结果
    print("\n" + "=" * 50)
    print("CLUSTERING RESULTS")
    print("=" * 50)

    ans = texts.llm.map(query)

    # 打印初始聚类的结果
    print("\nInitial Clusters:")
    for cluster_id, indices in initial_clusters.items():
        cluster_results = ans[indices].tolist()
        true_count = sum(1 for x in cluster_results if x)
        total = len(cluster_results)
        mean = true_count / total
        print(
            f"Cluster {cluster_id}: size={total}, mean={mean:.4f} "
            f"({true_count} positive / {total-true_count} negative)"
        )

    # 打印调整后聚类的结果
    print("\nAdjusted Clusters:")
    for cluster_id, indices in adjusted_clusters.items():
        cluster_results = ans[indices].tolist()
        true_count = sum(1 for x in cluster_results if x)
        total = len(cluster_results)
        mean = true_count / total
        print(
            f"Cluster {cluster_id}: size={total}, mean={mean:.4f} "
            f"({true_count} positive / {total-true_count} negative)"
        )

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
