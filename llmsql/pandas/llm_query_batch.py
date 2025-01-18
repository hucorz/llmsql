import pandas as pd
import json
from ..llm import BatchLLM
from llmsql.utils import get_ordered_columns, parse_template


# @pd.api.extensions.register_series_accessor("batch_llm")
@pd.api.extensions.register_dataframe_accessor("batch_llm")
class BatchLLMQuery:
    """Pandas accessor for LLM operations"""

    _llm: BatchLLM = None

    def __init__(self, pandas_obj):
        if self._llm is None:
            raise ValueError("Please configure LLM estimator first using llmsql.init()")

        self._obj = pandas_obj

    def __call__(self, query: str, batch_size: int = 4) -> pd.Series | pd.DataFrame:
        input_fields, output_fields = parse_template(query)
        input_fields = get_ordered_columns(self._obj, input_fields)
        df = self._obj[input_fields]
        # TODO: if merge data, does sorting row data work?
        # df = df.sort_values(by=input_fields)

        results = {field: [] for field in output_fields}
        for i in range(0, len(df), batch_size):
            batch = df[i : i + batch_size].to_dict("records")
            context = "\n".join(
                f"Data {i + 1}: {json.dumps(item)} " for i, item in enumerate(batch)
            )
            predictions = self._llm(query=query, context=context, batch_size=len(batch))
            for field in output_fields:
                results[field].extend(getattr(predictions, field))

        return pd.DataFrame(results, index=self._obj.index)
