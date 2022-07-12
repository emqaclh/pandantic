"""
Validators for data validation and amendment.
"""
from typing import Callable, Optional, Literal, Tuple, List
from numbers import Number

import pandas as pd
import numpy as np

from src import columns


class Validator:

    column_types = [columns.ObjectColumn]
    mandatory = False
    description = "Empty validation."
    amendment = None

    def __init__(self, mandatory: Optional[bool], description: Optional[str]) -> None:
        self.mandatory = mandatory if mandatory is not None else False
        if description is not None:
            self.description = description

    def evaluate(self, series) -> Tuple[pd.Series, int, bool]:
        if not isinstance(series, pd.Series):
            raise TypeError("A pandas.Series object must be provided")

        issue_count, valid = self._evaluate(series)
        if not valid:
            series = self.amendment(series)
            issue_count, valid = self._evaluate(series)

        return series, issue_count, valid

    # pylint: disable=unused-argument
    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:
        return 0, True

    def amend(self, amendment: Callable[[pd.Series], pd.Series]):
        self.amendment = amendment


class RangeValidator(Validator):

    column_types = [columns.NumberColumn]

    def __init__(
        self,
        mandatory: Optional[bool],
        description: Optional[str],
        min_value: Number,
        max_value: Number,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> None:
        super().__init__(mandatory, description)

        if min_value is None or max_value is None:
            raise ValueError("min_value and max_value must be provided.")

        self.inclusive = inclusive
        self.min_value, self.max_value = min_value, max_value

    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:
        result = False
        non_null = series.count()

        if np.isinf(self.max_value):
            if self.inclusive == "left" or self.inclusive == "both":
                result = series.ge(self.min_value)
            else:
                result = series.gt(self.min_value)

        elif np.isneginf(self.min_value):
            if self.inclusive == "right" or self.inclusive == "both":
                result = series.le(self.min_value)
            else:
                result = series.lt(self.min_value)

        else:
            result = series.between(
                self.min_value, self.max_value, inclusive=self.inclusive
            )

        result = result.sum()

        return (non_null - result), not ((non_null - result) > 0)


class CategoriesValidator(Validator):
    def __init__(
        self, mandatory: Optional[bool], description: Optional[str], categories: List
    ) -> None:
        super().__init__(mandatory, description)
        if not len(categories):
            raise ValueError("Categories list cannot be empty.")

        self.categories = categories

    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:
        non_null = series.count()

        in_category = series.isin(self.categories).sum()

        return (non_null - in_category), not ((non_null - in_category) > 0)
