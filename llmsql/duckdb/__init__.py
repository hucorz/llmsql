import json
import re
import duckdb
from duckdb import DuckDBPyConnection
from duckdb.typing import VARCHAR
from llmsql.duckdb.rewriter import rewrite_sql
from llmsql.duckdb.udf import (
    llm_udf,
    llm_udf_filter,
)

# if REGISTERED_MODEL is None:
#     raise RuntimeError("Call llmsql.init before importing from llmsql.duckdb")

# Override duckdb.sql(...)
original_sql_fn = duckdb.sql


def override_sql(sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_sql_fn(sql_query)


duckdb.sql = override_sql
duckdb.create_function("LLM", llm_udf, return_type=VARCHAR, type="arrow")
duckdb.create_function("LLM_FILTER", llm_udf_filter, return_type=VARCHAR, type="arrow")


# Override duckdb.connect(...); conn.execute
original_connect = duckdb.connect
original_execute = duckdb.DuckDBPyConnection.execute
original_connection_sql = duckdb.DuckDBPyConnection.sql


def override_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connection.create_function("LLM", llm_udf, return_type=VARCHAR, type="arrow")
    connection.create_function("LLM_FILTER", llm_udf_filter, return_type=VARCHAR, type="arrow")
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
