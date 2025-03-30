import sys

sys.path.append("/fs/fast/u20247643/llmsql")

import json
import re
import random
import duckdb
from duckdb import DuckDBPyConnection

from llmsql.duckdb.prompts import DECOMPOSITION_SYSTEM_PROMPT


def wrap_sql_to_extract_json_object(sql_query: str, output_fields: list[tuple[str, str]]) -> str:
    output_fields_sql = ", ".join(
        [
            (
                field
                if function_type == "normal"
                else f"json_extract_string(json_output, '$.{field}') AS {field}"
            )
            for field, function_type in output_fields
        ]
    )
    return f"SELECT\n{output_fields_sql}\nFROM ({sql_query});"


def merge_select_llm_expressions(llm_expressions: list[str], output_fields: list[str]) -> str:
    # Extract the content inside LLM() for each expression
    query_fields, data_fields = [], []
    for expr in llm_expressions:
        content = expr[4:-1]  # Remove 'LLM(' and ')'
        # query may contains "," so use regex to extract
        query_field = re.search(r"\'(.*?)\'", content).group(1)
        content = content.replace(f"'{query_field}'", "")  # Remove the query field
        query_fields.append(query_field.strip())
        data_fields.extend([field.strip() for field in content.split(",")[1:]])

    query_parts, output_types = [], []

    for query_field in query_fields:
        if query_field.startswith("'") and query_field.endswith("'"):
            query_field = query_field[1:-1]
        if "->" in query_field:
            parts = query_field.split("->")
            query, output_format = parts[0].strip(), parts[1].strip()
            query_parts.append(query)
            output_types.append(output_format)

    assert query_parts and output_types
    assert len(query_parts) == len(output_types)
    assert len(output_types) == len(output_fields)

    # Combine queries with Q1, Q2, etc. prefixes
    merged_query = " ".join([f"Q{i+1}: {query_parts[i]}" for i in range(len(query_parts))])
    merged_output_format = ", ".join(
        [f"{{{output_fields[i]}: {output_types[i]}}}" for i in range(len(output_types))]
    )
    merged_data_fields = ", ".join(data_fields)

    return f"""LLM('{merged_query} -> {merged_output_format}', {merged_data_fields})"""


def segment_select_clause(select_clause: str) -> str:
    select_clause = select_clause[6:]

    udf_pattern = r"(\w+\([^)]*\))\s+AS\s+(\w+)"  # 匹配 UDF 函数及别名
    udf_matches = re.findall(udf_pattern, select_clause)

    fields = []
    for match in udf_matches:
        udf_function, alias = match
        fields.append(f"{udf_function} AS {alias}")

    sql_query_no_udf = re.sub(udf_pattern, "", select_clause)
    sql_query_no_select = sql_query_no_udf.replace("SELECT", "").strip()
    fields.extend([field.strip() for field in sql_query_no_select.split(",")])
    fields = [field for field in fields if field]
    return fields


def rewrite_select_clause(select_clause: str) -> str:
    # get raw output fields
    raw_clause_segment = segment_select_clause(select_clause)
    raw_function_types = ["llm" if "LLM('" in field else "normal" for field in raw_clause_segment]
    raw_output_fields = [
        field if "AS" not in field else field.split("AS")[1].strip() for field in raw_clause_segment
    ]
    raw_output_fields = list(zip(raw_output_fields, raw_function_types))

    # rewrite LLM expressions
    select_llm_matches = list(re.finditer(r"LLM\(.*?\)(\s+AS\s+\w+)?", select_clause))

    if len(select_llm_matches) > 1:
        llm_expressions = []
        output_fields = []

        for match in select_llm_matches:
            full_match = match.group(0)
            llm_expr = re.search(r"LLM\(.*?\)", full_match).group(0)
            llm_expressions.append(llm_expr)

            # Check for AS clause
            as_clause = match.group(1)
            if as_clause:
                output_field = re.search(r"AS\s+(\w+)", as_clause, re.IGNORECASE).group(1)
                output_fields.append(output_field)
            else:
                output_fields.append("")

        merged_llm_expressions = merge_select_llm_expressions(llm_expressions, output_fields)
        merged_llm_expressions = f"{merged_llm_expressions} AS json_output"

        # Remove all LLM expressions from the select clause
        patterns_to_remove = []
        for match in select_llm_matches:
            escaped_match = re.escape(match.group(0))
            patterns_to_remove.append(f"{escaped_match},\\s*")
            patterns_to_remove.append(f"\\s*,{escaped_match}")
            patterns_to_remove.append(escaped_match)

        modified_select = select_clause
        for pattern in patterns_to_remove:
            modified_select = re.sub(r"LLM\(.*?\)(\s+AS\s+\w+)?", "", modified_select)

        # Clean up the select clause
        select_items = [item.strip() for item in modified_select.split(",")]
        select_items = [item for item in select_items if item]

        # Rebuild the select clause with proper commas
        clean_select = ", ".join(select_items)
        if len(clean_select) == len("SELECT"):
            return f"{clean_select} {merged_llm_expressions}", raw_output_fields
        else:
            return f"{clean_select}, {merged_llm_expressions}", raw_output_fields
    else:
        return select_clause, raw_output_fields


def rewrite_where_clause(where_clause: str) -> str:
    user_prompt = f"input_where_clause: {where_clause}"
    messages = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    from llmsql.llm import APIModel

    response = APIModel().query(messages)

    llm_pattern = r"LLM\(\s*([\'])(.*?)\1.*?\)"

    def wrap_llm_in_boolean(match):
        full_match = match.group(0)
        # Wrap the LLM function in a boolean expression (= 'true')
        return f"{full_match} = 'true'"

    # Replace LLM functions with wrapped versions
    final_response = re.sub(llm_pattern, wrap_llm_in_boolean, response)
    final_response = final_response.replace("LLM", "LLM_FILTER")

    return final_response


def rewrite_sql(sql_query: str) -> str:
    # if not select clause, return original sql
    if not extract_sql_clause(sql_query, "SELECT"):
        return sql_query
    """Intercepts DuckDB SQL query string and outputs an updated query."""
    raw_select_clause = extract_sql_clause(sql_query, "SELECT")
    select_clause, output_fields = rewrite_select_clause(raw_select_clause)

    raw_where_clause = extract_sql_clause(sql_query, "WHERE")
    where_clause = rewrite_where_clause(raw_where_clause)

    rewritten_sql = sql_query.replace(raw_select_clause, select_clause)
    rewritten_sql = rewritten_sql.replace(raw_where_clause, where_clause)
    rewritten_sql = wrap_sql_to_extract_json_object(rewritten_sql, output_fields)

    return rewritten_sql


def extract_sql_clause(sql_query: str, clause_type: str) -> str:
    """
    Extracts a specific clause from a SQL query.

    This function matches only the specified clause and its content, stopping at
    subsequent SQL keywords that might follow the clause.

    Args:
        sql_query: The SQL query string to extract the clause from.
        clause_type: The type of clause to extract ('SELECT', 'WHERE', etc.)

    Returns:
        The extracted clause with its content, or an empty string if the clause is not found.
    """
    sql_upper = sql_query.upper()
    clause_upper = clause_type.upper()

    # Define patterns for different clause types
    patterns = {
        "SELECT": r"\bSELECT\b(.*?)(?:\b(?:FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
        "WHERE": r"\bWHERE\b(.*?)(?:\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT|FOR|WINDOW)\b|$)",
        "FROM": r"\bFROM\b(.*?)(?:\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
        "GROUP BY": r"\bGROUP\s+BY\b(.*?)(?:\b(?:HAVING|ORDER\s+BY|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
        "ORDER BY": r"\bORDER\s+BY\b(.*?)(?:\b(?:LIMIT|OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
        "HAVING": r"\bHAVING\b(.*?)(?:\b(?:ORDER\s+BY|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
        "LIMIT": r"\bLIMIT\b(.*?)(?:\b(?:OFFSET|UNION|INTERSECT|EXCEPT)\b|$)",
    }

    if clause_upper not in patterns:
        raise ValueError(
            f"Unsupported clause type: {clause_type}. Supported types are: {', '.join(patterns.keys())}"
        )

    pattern = patterns[clause_upper]
    match = re.search(pattern, sql_upper, re.DOTALL | re.IGNORECASE)

    if match:
        clause_pos = match.start()
        end_pos = match.end(1)
        clause_len = len(clause_upper)

        # Extract the clause from the original SQL (preserving case)
        extracted_clause = (
            clause_type + sql_query[clause_pos + clause_len : clause_pos + end_pos - clause_pos]
        )
        extracted_clause = extracted_clause.strip()
        return extracted_clause

    return ""


if __name__ == "__main__":
    origin_sql = """
SELECT
  update_date,
  LLM('Analyze {abstract} for AI relevance? -> bool', p.abstract) AS is_AI,
  LLM('Check {title} for math terminology? -> bool', p.title) AS has_math
FROM papers as p
WHERE LLM("{update_date} 时间为 2000-2009的论文的 {abstract} 是否和 AI 相关? -> bool", p.date, p.abstract)
"""
    print(rewrite_sql(origin_sql))
