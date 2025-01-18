from typing import Optional, Type, Literal
from llmsql.utils import parse_template
import dspy


class BaseDynamicSignature(dspy.Signature):
    """You are analyzing multiple data records to answer the user's question.
    The data contains {count} records that you need to process.
    For each record in the data, you should:
    1. Understand the user's query and what information they are looking for
    2. Analyze the record's content in relation to the query
    3. Generate appropriate response fields for this record
    4. Move to the next record and repeat the process

    Remember to maintain consistency in your analysis across all records.
    Your output should contain a list of results, one for each record in the data.
    """

    query: str = dspy.InputField(desc="user's query")
    context: str = dspy.InputField(desc="one or more datas to answer user's query")
    count: int = dspy.InputField(desc="number of responses to generate")

    @classmethod
    def get_input_fields(cls) -> dict[str, tuple[Type, dspy.InputField]]:
        """Get all input fields with their types and field definitions.

        Returns:
            A dictionary mapping field names to tuples of (type, field),
            where type is the annotation type and field is the InputField instance.
        """
        return {
            name: (cls.__annotations__[name], field)
            for name, field in cls.__dict__.items()
            if hasattr(field, "_field_type")
            and getattr(field, "_field_type") == "input"
            and name in cls.__annotations__
        }


def create_dynamic_signature(output_fields: dict[str, str]) -> Type:
    """Create a dynamic signature class based on output fields"""
    base_attrs = BaseDynamicSignature.get_input_fields()

    attrs = {
        "__annotations__": {
            **{name: type_ for name, (type_, _) in base_attrs.items()},
            **{name: eval(f"list[{type_}]") for name, type_ in output_fields.items()},
        },
        **{name: field for name, (_, field) in base_attrs.items()},
        **{name: dspy.OutputField() for name in output_fields},
        "__doc__": BaseDynamicSignature.__doc__,
    }

    return type("DynamicSignature", (BaseDynamicSignature,), attrs)


class BatchLLM:
    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.lm = dspy.LM(model=model, api_base=api_base, api_key=api_key, **kwargs)
        dspy.settings.configure(lm=self.lm)
        self._predictor_cache: dict[str, tuple[dict[str, str], dspy.Predict]] = {}

    def _parse_template(self, query: str) -> tuple[list[str], dict[str, str], Type]:
        """Convert template query to DSPy signature format

        Example:
            "Is {col1} same as {col2} -> {col3}, {col4:type4}"
            Input fields (before ->) cannot have type annotations
            Output fields (after ->) can optionally have type annotations
        """
        input_fields, output_fields = parse_template(query)
        dynamic_signature = create_dynamic_signature(output_fields)

        return input_fields, output_fields, dynamic_signature

    def _get_predictor(self, query: str):
        """Get or create predictor for query template"""
        if query not in self._predictor_cache:
            _, output_fields, dynamic_signature = self._parse_template(query)
            self._predictor_cache[query] = (output_fields, dspy.Predict(dynamic_signature))
        return self._predictor_cache[query]

    def __call__(self, query: str, context: str, batch_size: int):
        output_fields, predictor = self._get_predictor(query)
        prediction_obj = predictor(query=query, context=context, count=batch_size)

        # check output fields
        assert all(
            field_name in prediction_obj for field_name in output_fields
        ), "Missing output fields"

        # check length
        expected_length = len(getattr(prediction_obj, next(iter(output_fields))))
        if not all(
            len(getattr(prediction_obj, field_name)) == expected_length
            for field_name in output_fields
        ):
            raise ValueError("All output fields must have the same length")

        return prediction_obj
