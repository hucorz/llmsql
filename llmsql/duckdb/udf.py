import re
import json
import random
import pandas as pd
import pyarrow as pa
from llmsql import REGISTERED_MODEL

assert REGISTERED_MODEL, "REGISTERED_MODEL must be valid"


def llm_udf(query: pa.Array, *args) -> str:
    query = query.to_pylist()
    args = [arg.to_pylist() for arg in args]
    query = str(query[0])
    query_part, output_format, fields = parse_query(query)
    data = {field: arg for field, arg in zip(fields, args)}
    data = pd.DataFrame(data).to_dict(orient="records")
    output = REGISTERED_MODEL.execute(data=data, query=query_part, output_format=output_format)
    output = [json.dumps(item) for item in output]
    return pa.array(output)


def llm_udf_filter(query: str, *args) -> bool:
    query = query.to_pylist()
    args = [arg.to_pylist() for arg in args]
    query = str(query[0])
    query_part, output_format, fields = parse_query(query)
    data = {field: arg for field, arg in zip(fields, args)}
    data = pd.DataFrame(data).to_dict(orient="records")
    output = REGISTERED_MODEL.execute(
        data=data, query=query_part, output_format="{true_or_false: bool}"
    )
    output = [list(item.values())[0] for item in output]
    # TODO output here is a list of 'true' or 'false', but it works
    # duckdb will transform to bool automatically?
    return pa.array(output)


def parse_query(query: str):
    query_part, format_part = query.split("->")
    query_part, format_part = query_part.strip(), format_part.strip()
    fields_pattern = r"\{(\w+)\}"
    fields = re.findall(fields_pattern, query_part)
    fields = [field.strip() for field in fields]
    return query_part, format_part, fields
