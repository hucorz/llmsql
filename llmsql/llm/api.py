import os
from openai import OpenAI


class APIModel:
    def __init__(self, base_url: str = "https://api.deepseek.com"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        assert api_key, "DEEPSEEK_API_KEY must be set"
        self.client = OpenAI(
            api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1/")
        # self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict[str, str]]):
        response = self.client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            # model="Pro/deepseek-ai/DeepSeek-V3",
            messages=messages,
            stream=False,
            temperature=0.0,
        )
        return response.choices[0].message.content
