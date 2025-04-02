import re
import json
import traceback
import random
import pandas as pd
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
from llmsql import GlobalEntryPoint

assert GlobalEntryPoint, "GlobalEntryPoint must be valid"


def llm_udf(query: pa.Array, *args) -> str:
    query = str(query.to_pylist()[0])
    args = [arg.to_pylist() for arg in args]
    query_part, output_format, fields = parse_query(query)
    data = {field: arg for field, arg in zip(fields, args)}
    data = pd.DataFrame(data).to_dict(orient="records")
    output = GlobalEntryPoint.query(
        data=data,
        query=query_part,
        output_format=output_format,
        use_turbo=True,
        use_cache=True,
        is_full_data=False,
    )
    return pa.array(output)


def llm_udf_filter(query: pa.Array, *args) -> bool:
    query = str(query.to_pylist()[0])
    args = [arg.to_pylist() for arg in args]
    query_part, output_format, fields = parse_query(query)
    data = {field: arg for field, arg in zip(fields, args)}
    data = pd.DataFrame(data).to_dict(orient="records")
    output = GlobalEntryPoint.query(
        data=data, query=query_part, output_format="{true_or_false: bool}"
    )
    output = [list(json.loads(item).values())[0] for item in output]
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
