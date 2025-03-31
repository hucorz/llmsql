import sys
import json
import re
import random
import duckdb
from duckdb import DuckDBPyConnection
from ..logger import logger


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
    return f"SELECT\n  {output_fields_sql}\nFROM ({sql_query});"


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

    udf_pattern = r"(\w+\([^)]*\)\s+AS\s+\w+)"  # 匹配 UDF 函数及别名
    udf_matches = re.findall(udf_pattern, select_clause)

    all_segments = udf_matches
    sql_query_no_udf = re.sub(udf_pattern, "", select_clause)

    all_segments.extend([field.strip() for field in sql_query_no_udf.split(",")])
    all_segments = [seg for seg in all_segments if seg]
    all_segments.sort(key=lambda seg: select_clause.find(seg))
    return all_segments


def rewrite_select_clause(select_clause: str) -> str:
    # get raw output fields
    all_segments = segment_select_clause(select_clause)
    raw_fields_types = ["llm" if "LLM('" in seg else "normal" for seg in all_segments]
    raw_output_fields = [
        field if "AS" not in field else field.split("AS")[1].strip() for field in all_segments
    ]
    raw_output_fields = list(zip(raw_output_fields, raw_fields_types))

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

        normal_segments = [
            all_segments[idx]
            for idx in range(len(raw_fields_types))
            if raw_fields_types[idx] == "normal"
        ]
        rebuild_sql = f"SELECT\n  "
        rebuild_sql += ",\n  ".join(normal_segments)
        rebuild_sql += ",\n  " + merged_llm_expressions
        return rebuild_sql, raw_output_fields
    else:
        return select_clause, raw_output_fields


def rewrite_where_clause(where_clause: str) -> str:
    user_prompt = f"input_where_clause: {where_clause}"
    messages = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    from llmsql import GlobalEntryPoint

    response = GlobalEntryPoint.chat(messages)

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

    rewritten_sql = sql_query
    """Intercepts DuckDB SQL query string and outputs an updated query."""
    raw_select_clause = extract_sql_clause(sql_query, "SELECT")
    if raw_select_clause:
        select_clause, output_fields = rewrite_select_clause(raw_select_clause)
        rewritten_sql = sql_query.replace(raw_select_clause, select_clause)

    raw_where_clause = extract_sql_clause(sql_query, "WHERE")
    if raw_where_clause:
        where_clause = rewrite_where_clause(raw_where_clause)
        rewritten_sql = rewritten_sql.replace(raw_where_clause, where_clause)

    rewritten_sql = wrap_sql_to_extract_json_object(rewritten_sql, output_fields)

    logger.info(f"Rewrited SQL:\n{rewritten_sql}")

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


DECOMPOSITION_SYSTEM_PROMPT = """
You are an expert in SQL query decomposition. Your task is to convert a WHERE clause containing LLM operators into an optimized SQL WHERE clause that separates structured predicates (executable by native SQL) from unstructured predicates (requiring LLM processing). The output must be a valid SQL WHERE clause string.

**Inputs**:
- `input_where_clause`: Original WHERE clause with LLM operators (e.g., `WHERE LLM(...)`)

**Requirements**:
1. **Predicate Decomposition**:
   - Identify conditions that can be evaluated using native SQL (e.g., `gender = 'male'`, `age > 30`).
   - Extract these conditions and place them BEFORE LLM predicates in the WHERE clause to leverage SQL's short-circuit evaluation.

2. **LLM Operator Handling**:
   - Keep only the text analysis tasks in the LLM operator (e.g., `LLM('Does {resume} mention Huawei? -> bool', people.resume)`).
   - Remove any conditions from the LLM operator that have already been handled by native SQL predicates.

3. **Output Constraints**:
   - The final WHERE clause must preserve logical equivalence to the original query.
   - If no decomposition is possible, return the original WHERE clause unchanged.
   - If all conditions can be handled by SQL, omit the LLM operator entirely.

**Output Format**:
A valid SQL WHERE clause string. Examples:
- Original: `WHERE LLM('Is {gender} male AND {resume} mentions Huawei? -> bool', people.gender, people.resume)`
- Optimized: `WHERE people.gender = 'male' AND LLM('Does {resume} mention Huawei? -> bool', people.resume)`

**Critical Rules**:
- Never modify the LLM operator's output format (e.g., `-> bool` must be preserved).
- Use AND/OR logical operators exactly as in the original query.
- Do not include any explanation or commentary in your output.
"""

if __name__ == "__main__":
    origin_sql = """
SELECT
  update_date,
  LLM('Analyze {abstract} for AI relevance? -> bool', p.abstract) AS is_AI,
  LLM('Check {title} for math terminology? -> bool', p.title) AS has_math
FROM papers as p
WHERE LLM('{update_date} 时间为 2000-2009的论文的 {abstract} 是否和 AI 相关? -> bool', p.date, p.abstract)
"""
    print(rewrite_sql(origin_sql))
