import re
import json
import requests
from openai import OpenAI
from typing import Dict, List, Optional
import pandas as pd


class KTransformers:
    def __init__(self, base_url: str = "http://localhost:10002"):
        """Initialize the KTransformers LLM client.

        Args:
            base_url: The base URL of the KTransformers server.
        """
        self.base_url = base_url.rstrip("/")
        self.query_endpoint = f"{self.base_url}/v1/data/query"
        self.chat_client = OpenAI(api_key="xxx", base_url="http://localhost:10002/v1")

    def query(
        self,
        data: list[dict],
        query: str,
        output_format: str,
        use_turbo: bool = True,
        use_cache: bool = True,
        is_full_data: bool = False,
    ) -> str:
        payload = {
            "data": data,
            "query": query,
            "output_format": output_format,
            "use_turbo": use_turbo,
            "use_cache": use_cache,
            "is_full_data": is_full_data,
        }

        response = requests.post(self.query_endpoint, json=payload)
        result = response.json().get("result", [])

        if result and isinstance(result, list) and len(result) > 0:
            return result

        return ""

    def chat(self, messages: list[dict[str, str]]):
        print(f"KTransformers chat")
        response = self.chat_client.chat.completions.create(
            model="xxx",
            messages=messages,
            stream=False,
            temperature=0.0,
        )
        return response.choices[0].message.content
