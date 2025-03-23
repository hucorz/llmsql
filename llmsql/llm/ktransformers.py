import re
import json
import requests
from typing import Dict, List, Optional, Union
import pandas as pd
from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM


class KTransformers(LLM):
    def __init__(self, base_url: str):
        """Initialize the KTransformers LLM client.

        Args:
            base_url: The base URL of the KTransformers server.
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/data/query"

    def execute(
        self,
        df: pd.DataFrame,
        query: str,
        output_format: str,
        use_turbo: bool = True,
        use_cache: bool = True,
        is_full_data: bool = False,
    ) -> str:
        """Execute a single query against the KTransformers server.

        Args:
            df: The DataFrame to query.
            query: The user query for the LLM call.
            output_format: The output format for the response.
            use_turbo: Whether to use turbo mode for processing.
            use_cache: Whether to use caching.
            is_full_data: Whether to use full data for processing.

        Returns:
            The response from the KTransformers server.
        """
        # 使用正则提取 query 中涉及到的列，在 '{}' 中
        fields = re.findall(r"\{([^}]*)\}", query)
        fields = [field.strip() for field in fields]
        df = df[fields].to_dict(orient="records")

        payload = {
            "data": df,
            "query": query,
            "output_format": output_format,
            "use_turbo": use_turbo,
            "use_cache": use_cache,
            "is_full_data": is_full_data,
        }

        response = requests.post(self.endpoint, json=payload)
        result = response.json().get("result", [])

        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("result", "")

        return ""

    def execute_batch(
        self,
        fields: List[Dict[str, str]],
        query: str,
        data_path: str = None,
        use_turbo: bool = True,
        use_cache: bool = True,
        is_full_data: bool = False,
        stride: int = 1,
        user_format: str = "{result: string}",
    ) -> List[str]:
        """Execute a batch query against the KTransformers server.

        Args:
            fields: A list of dicts mapping from column names to values.
            query: The user query for the LLM call.
            data_path: Path to the data file that will be used for queries.
            use_turbo: Whether to use turbo mode for processing.
            use_cache: Whether to use caching.
            is_full_data: Whether to use full data for processing.
            stride: The stride to use for processing data.
            user_format: The format for the response.

        Returns:
            A list of responses from the KTransformers server.
        """
        # If the batch is empty, return an empty list
        if not fields:
            return []

        # Call execute for each item in the batch
        return [
            self.execute(
                fields=field,
                query=query,
                data_path=data_path,
                use_turbo=use_turbo,
                use_cache=use_cache,
                is_full_data=is_full_data,
                stride=stride,
                user_format=user_format,
            )
            for field in fields
        ]
