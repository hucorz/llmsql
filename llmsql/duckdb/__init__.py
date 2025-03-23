import json
import re
import random
import duckdb
from duckdb import DuckDBPyConnection

from llmsql import REGISTERED_MODEL, REGISTERED_API_MODEL
from llmsql.duckdb.prompts import DECOMPOSITION_SYSTEM_PROMPT

# if REGISTERED_MODEL is None:
#     raise RuntimeError("Call llmsql.init before importing from llmsql.duckdb")


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


def get_query_fields(query: str):
    matches = re.findall(r"\{[^{}]*\}", query)
    return [match[1:-1] for match in matches]


def extend_llm_query_with_json_object(sql_query: str) -> str:
    # Regular expression to find LLM operators with only a string parameter
    llm_pattern = r'LLM\(\s*(["\'])(.*?)\1\s*\)'

    def process_match(match):
        # Extract the content inside LLM()
        quote_char = match.group(1)  # Either ' or "
        content = match.group(2)

        # Check if the content contains the arrow pattern
        if "->" in content:
            prompt_part, output_part = content.split("->", 1)
            prompt_part = prompt_part.strip()
            output_part = output_part.strip()

            # Extract fields from the prompt part (text inside curly braces)
            fields = re.findall(r"\{([^{}]*?)\}", prompt_part)

            # If fields are found, create a JSON_OBJECT with them
            if fields:
                json_object_parts = []
                for field in fields:
                    json_object_parts.append(f"'{field}', {field}")

                json_object_str = f"JSON_OBJECT({', '.join(json_object_parts)})"

                # Construct the new LLM operator with the same quote character
                return f"LLM({quote_char}{prompt_part} -> {output_part}{quote_char}, {json_object_str})"

        # If no arrow pattern or no fields found, return the original match
        return match.group(0)

    # Replace all LLM operators in the SQL query
    modified_sql = re.sub(llm_pattern, process_match, sql_query)

    return modified_sql


def wrap_sql_to_extract_json_object(sql_query: str, output_fields: list[tuple[str, str]]) -> str:
    output_fields_sql = ", ".join(
        [
            (
                field
                if field_type == "normal"
                else f"json_extract_string(json_output, '$.{field}') AS {field}"
            )
            for field, field_type in output_fields
        ]
    )
    return f"SELECT\n{output_fields_sql}\nFROM ({sql_query});"


def merge_select_llm_expressions(llm_expressions: list[str]) -> str:
    # Extract the content inside LLM() for each expression
    parsed_expressions = []
    for expr in llm_expressions:
        content = expr[4:-1]  # Remove 'LLM(' and ')'
        parsed_expressions.append(content)

    query_parts = []
    output_formats = []

    for expr_content in parsed_expressions:
        # Remove outer quotes if present
        if (expr_content.startswith('"') and expr_content.endswith('"')) or (
            expr_content.startswith("'") and expr_content.endswith("'")
        ):
            expr_content = expr_content[1:-1]

        # Split on the arrow
        if "->" in expr_content:
            parts = expr_content.split("->")
            query = parts[0].strip()
            output_format = parts[1].strip()
            query_parts.append(query)
            output_formats.append(output_format)

    if not query_parts or not output_formats:
        return llm_expressions[0] if llm_expressions else ""

    # Combine queries with Q1, Q2, etc. prefixes
    merged_query = " ".join([f"Q{i+1} {query_parts[i]}" for i in range(len(query_parts))])
    merged_output_format = ", ".join(output_formats)

    return extend_llm_query_with_json_object(f"""LLM('{merged_query} -> {merged_output_format}')""")


def rewrite_select_clause(select_clause: str) -> str:
    raw_output_fields = [field.strip() for field in select_clause[6:].split(",")]
    raw_output_types = ["llm" if "LLM('" in field else "normal" for field in raw_output_fields]
    raw_output_fields = [
        field if "AS" not in field else field.split("AS")[1].strip() for field in raw_output_fields
    ]
    raw_output_fields = list(zip(raw_output_fields, raw_output_types))

    # Define the regular expression pattern to match the LLM expression
    llm_pattern = r"LLM\(.*?\)(\s+AS\s+\w+)?"

    select_llm_matches = list(re.finditer(llm_pattern, select_clause))

    if len(select_llm_matches) > 1:
        # Extract each LLM expression and its output field
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

        merged_llm_expressions = merge_select_llm_expressions(llm_expressions)
        merged_llm_expressions = f"{merged_llm_expressions} AS json_output"

        # Create a list of all LLM expressions to remove
        patterns_to_remove = []
        for match in select_llm_matches:
            escaped_match = re.escape(match.group(0))
            patterns_to_remove.append(f"{escaped_match},\\s*")
            patterns_to_remove.append(f"\\s*,{escaped_match}")
            patterns_to_remove.append(escaped_match)

        # Remove all LLM expressions from the select clause
        modified_select = select_clause
        for pattern in patterns_to_remove:
            modified_select = re.sub(pattern, "", modified_select)

        # print(f"Modified select clause:\n{modified_select}\n")

        # Clean up the select clause
        # Split by commas and filter out empty items
        select_items = [item.strip() for item in modified_select.split(",")]
        select_items = [item for item in select_items if item]
        # print(f"Select items:\n{select_items}\n")

        # Rebuild the select clause with proper commas
        if select_items:
            clean_select = ", ".join(select_items)
            if len(clean_select) == len("SELECT"):
                return f"{clean_select} {merged_llm_expressions}", raw_output_fields
            else:
                return f"{clean_select}, {merged_llm_expressions}", raw_output_fields
        else:
            # Otherwise, add the merged expression with a comma
            return f"{select_clause}, {merged_llm_expressions}", raw_output_fields
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

    # print(f"Decomposed WHERE clause:\n{response}\n")

    # Convert double-quoted LLM calls to single-quoted LLM calls
    # Find all LLM("...") patterns and replace with LLM('...')
    double_quote_pattern = r'LLM\(\s*"(.+?)"\s*\)'
    response = re.sub(double_quote_pattern, lambda m: f"LLM('{m.group(1)}')", response)

    # First, apply the JSON_OBJECT transformation
    modified_response = extend_llm_query_with_json_object(response)

    # Then, check if there's an LLM function in the WHERE clause and wrap it in a boolean expression
    llm_pattern = r"LLM\(\s*([\'])(.*?)\1\s*,\s*JSON_OBJECT\((.*?)\)\s*\)"

    def wrap_llm_in_boolean(match):
        full_match = match.group(0)
        # Wrap the LLM function in a boolean expression (= 'true')
        return f"{full_match} = 'true'"

    # Replace LLM functions with wrapped versions
    final_response = re.sub(llm_pattern, wrap_llm_in_boolean, modified_response)
    final_response = final_response.replace("LLM", "LLM_FILTER")

    return final_response


def rewrite_sql(sql_query: str) -> str:
    # return sql_query
    # if not select clause, return original sql
    if not extract_sql_clause(sql_query, "SELECT"):
        return sql_query
    """Intercepts DuckDB SQL query string and outputs an updated query."""
    raw_select_clause = extract_sql_clause(sql_query, "SELECT")
    select_clause, output_fields = rewrite_select_clause(raw_select_clause)
    # print(f"Rewritten select clause:\n{select_clause}\n")
    raw_where_clause = extract_sql_clause(sql_query, "WHERE")
    # print(f"Raw where clause:\n{raw_where_clause}\n")
    where_clause = rewrite_where_clause(raw_where_clause)
    # print(f"Rewritten where clause:\n{where_clause}\n")

    rewritten_sql = sql_query.replace(raw_select_clause, select_clause)
    rewritten_sql = rewritten_sql.replace(raw_where_clause, where_clause)
    rewritten_sql = wrap_sql_to_extract_json_object(rewritten_sql, output_fields)

    print(f"Rewritten SQL:\n{rewritten_sql}\n")
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


# Override duckdb.sql(...)
original_sql_fn = duckdb.sql


def override_sql(sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_sql_fn(sql_query)


duckdb.sql = override_sql
duckdb.create_function("LLM", llm_udf)
duckdb.create_function("LLM_FILTER", llm_udf_filter)


# Override duckdb.connect(...); conn.execute
original_connect = duckdb.connect
original_execute = duckdb.DuckDBPyConnection.execute
original_connection_sql = duckdb.DuckDBPyConnection.sql


def override_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connection.create_function("LLM", llm_udf)
    connection.create_function("LLM_FILTER", llm_udf_filter)
    return connection


def override_execute(self, sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_execute(self, sql_query)


def override_connect_sql(self, sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_connection_sql(self, sql_query)


duckdb.connect = override_connect
duckdb.DuckDBPyConnection.execute = override_execute
duckdb.DuckDBPyConnection.sql = override_connect_sql
