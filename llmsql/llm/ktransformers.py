import re
import json
import requests
from typing import Dict, List, Optional
import pandas as pd


class KTransformers:
    def __init__(self, base_url: str = "http://localhost:10002"):
        """Initialize the KTransformers LLM client.

        Args:
            base_url: The base URL of the KTransformers server.
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/data/query"

    def execute(
        self,
        data: list[dict],
        query: str,
        output_format: str,
        use_turbo: bool = True,
        use_cache: bool = True,
        is_full_data: bool = False,
    ) -> str:
        """Execute a single query against the KTransformers server.

        Args:
            data: The data to query.
            query: The user query for the LLM call.
            output_format: The output format for the response.
            use_turbo: Whether to use turbo mode for processing.
            use_cache: Whether to use caching.
            is_full_data: Whether to use full data for processing.

        Returns:
            The response from the KTransformers server.
        """
        payload = {
            "data": data,
            "query": query,
            "output_format": output_format,
            "use_turbo": use_turbo,
            "use_cache": use_cache,
            "is_full_data": is_full_data,
        }

        response = requests.post(self.endpoint, json=payload)
        result = response.json().get("result", [])

        if result and isinstance(result, list) and len(result) > 0:
            return result

        return ""
