import pandas as pd
import re

INPUT_PATTERN = re.compile(r"\{([^}]+)\}")
OUTPUT_PATTERN = re.compile(r"\{([^:}]+)(?::([^}]+))?\}")


def parse_template(query: str) -> tuple[list[str], dict[str, str]]:
    """Convert template query to input and output fields

    Example:
        "Is {col1} same as {col2} -> {col3}, {col4:type4}"
        Input fields (before ->) cannot have type annotations
        Output fields (after ->) can optionally have type annotations
    """
    # Split into input and output parts
    parts = query.split("->")
    input_part = parts[0].strip()
    output_part = parts[1].strip() if len(parts) > 1 else ""

    # Parse input fields - no type annotations allowed
    input_fields = []  # list[str]
    for match in re.finditer(INPUT_PATTERN, input_part):
        field_name = match.group(1).strip()
        if ":" in field_name:
            raise ValueError(f"Input field '{field_name}' cannot have type annotation")
        input_fields.append(field_name)

    # Parse output fields - type annotations allowed
    output_fields = {}  # dict[str, str]
    if output_part:
        for match in re.finditer(OUTPUT_PATTERN, output_part):
            field_name = match.group(1).strip()
            field_type = match.group(2).strip() if match.group(2) else "str"
            output_fields[field_name] = field_type

    return input_fields, output_fields


## Reference:
# http://arxiv.org/abs/2403.05821
def get_field_score(df: pd.DataFrame, field: str):
    num_distinct = df[field].nunique(dropna=True)
    avg_length = df[field].apply(lambda x: len(str(x))).mean()
    return avg_length / num_distinct


def get_ordered_columns(df: pd.DataFrame, fields: list[str]):
    field_scores = {}

    for field in fields:
        field_scores[field] = get_field_score(df, field)

    reordered_fields = [
        field for field in sorted(fields, key=lambda field: field_scores[field], reverse=True)
    ]

    return reordered_fields
