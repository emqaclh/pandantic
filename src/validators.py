"""
Validators for data validation and amendment.
"""
from typing import Callable, Optional, Literal, Tuple, List, Union, Pattern
from numbers import Number

import pandas as pd
import numpy as np
import abc

from src import columns


class Validator(abc.ABC):

    column_types = [columns.ObjectColumn]
    mandatory = False
    description = "N/A"
    amendment = None

    def __init__(self, mandatory: Optional[bool], description: Optional[str]) -> None:
        self.mandatory = mandatory if mandatory is not None else False
        self.description = description if description is not None else "N/A"

    def evaluate(self, series) -> Tuple[pd.Series, int, int, bool, bool]:
        if not isinstance(series, pd.Series):
            raise TypeError("A pandas.Series object must be provided")

        original_issue_count, valid = self._evaluate(series)
        issue_count = original_issue_count
        amended = False

        if not valid and self.amendment is not None:
            series = self.amendment(series)
            issue_count, valid = self._evaluate(series)
            amended = True

        return series, original_issue_count, issue_count, valid, amended

    # pylint: disable=unused-argument
    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:
        raise NotImplementedError()

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
                result = series.le(self.max_value)
            else:
                result = series.lt(self.max_value)

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


class NonNullValidator(Validator):
    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:

        null_values = series.isnull().sum()

        return null_values, not (null_values)


class UniqueValidator(Validator):
    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:

        non_unique = series.duplicated(keep="first").sum()

        return non_unique, not (non_unique)


class PatternValidator(Validator):

    column_types = [columns.StringColumn]

    def __init__(
        self,
        mandatory: Optional[bool],
        description: Optional[str],
        pattern: Union[str, Pattern],
    ) -> None:
        super().__init__(mandatory, description)
        self.pattern = pattern

    def _evaluate(self, series: pd.Series) -> Tuple[int, bool]:

        non_null = series.count()
        match_count = series.str.fullmatch(self.pattern, case=True)

        return (non_null - match_count), not ((non_null - match_count) > 0)
