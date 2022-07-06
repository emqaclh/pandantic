from typing import Dict
from numpy import diag

import pandas as pd


class ObjectColumn:

    check_nulls = True
    check_unique = False

    def __init__(self, check_nulls=True, check_unique=False) -> None:
        self.check_nulls = check_nulls
        self.check_unique = check_unique

    def _coerce(self, series: pd.Series) -> pd.Series:
        return series

    def evaluate(self, series: pd.Series) -> Dict:

        if type(series) is not pd.Series:
            raise TypeError("A pandas.Series object must be provided")

        diagnostic = dict(coerced=False)

        valid_dtype = self.evaluate_dtype(series)
        if not valid_dtype:
            series = self._coerce(series)
            diagnostic['coerced'] = True

        valid_dtype = self.evaluate_dtype(series)
        if not valid_dtype:
            diagnostic['valid_dtype'] = False

        if not self.check_nulls:
            nulls = self.evaluate_nulls(series)
            diagnostic["nulls"] = nulls

        if self.check_unique:
            uniqueness = self.evaluate_uniqueness(series)
            diagnostic["uniqueness"] = uniqueness

        return series, diagnostic

    def evaluate_dtype(series: pd.Series) -> bool:
        return True

    def evaluate_nulls(self, series: pd.Series, return_count=False) -> bool:
        nulls_count = series.isnull().sum()

        if return_count:
            return nulls_count == 0, nulls_count

        return nulls_count == 0

    def evaluate_uniqueness(self, series: pd.Series) -> bool:
        return series.is_unique


class NumberColumn(ObjectColumn):
    pass


class IntColumn(NumberColumn):
    pass


class FloatColumn(NumberColumn):
    pass


class StringColumn(ObjectColumn):
    pass


class CategoryColumn(ObjectColumn):
    pass


class DatetimeColumn(ObjectColumn):
    pass


class TimedeltaColumn(ObjectColumn):
    pass
