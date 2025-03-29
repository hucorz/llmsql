import json


def llm_udf(prompt: str, contextargs: str) -> str:
    print("In llm_udf")
    print(f"Prompt: {prompt}")
    print(f"Contextargs: {contextargs}")
    fields = json.loads(contextargs)
    # output = REGISTERED_MODEL.execute(fields=fields, query=prompt)
    return r"""{"is_AI": true, "has_math": false}"""


def llm_udf_filter(prompt: str, contextargs: str) -> bool:
    print("In llm_udf_filter")
    print(f"Prompt: {prompt}")
    print(f"Contextargs: {contextargs}")
    fields = json.loads(contextargs)
    # 随机返回 true or false
    return random.choice([True, False])
    # output = REGISTERED_MODEL.execute(fields=fields, query=prompt)
