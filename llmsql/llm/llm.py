import re
from typing import Optional

import dspy

PATTERN = re.compile(r"\{([^:}]+)(?::([^}]+))?\}")


class LLM:
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

    def _parse_template(self, query: str) -> tuple[dict[str, str], str]:
        """Convert template query to DSPy signature format"""
        fields = {}
        for match in re.finditer(PATTERN, query):
            field_name = match.group(1).strip()
            # Default to str if type is not specified
            field_type = match.group(2).strip() if match.group(2) else "str"
            fields[field_name] = field_type

        field_signatures = [f"{name}: {type_}" for name, type_ in fields.items()]
        signature = f"query: str, context: str -> {', '.join(field_signatures)}"

        return fields, signature

    def _get_predictor(self, query: str):
        """Get or create predictor for query template"""
        if query not in self._predictor_cache:
            fields, signature = self._parse_template(query)
            self._predictor_cache[query] = (fields, dspy.ChainOfThought(signature))
        return self._predictor_cache[query]

    def __call__(self, query: str, context: str):
        """
        Execute query on context

        Args:
            query: Template query string (e.g., "tell me the {title: str} of the movie")
            context: Text to analyze

        Returns:
            Dict[str, Any]: Extracted fields and their values
        """
        fields, predictor = self._get_predictor(query)
        prediction = predictor(query=query, context=context)

        if len(fields) == 1:
            return getattr(prediction, next(iter(fields)))

        return {field_name: getattr(prediction, field_name) for field_name in fields}
