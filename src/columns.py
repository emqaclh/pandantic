from typing import Dict, Union, Tuple

import pandas as pd
import numpy as np


class ObjectColumn:

    check_nulls = False
    check_unique = False

    def __init__(self, check_nulls=False, check_unique=False) -> None:
        self.check_nulls = check_nulls
        self.check_unique = check_unique

    def coerce(self, series: pd.Series) -> pd.Series:
        return series

    def evaluate(self, series: pd.Series) -> Dict:

        if type(series) is not pd.Series:
            raise TypeError("A pandas.Series object must be provided")

        diagnostic = dict(coerced=False, warnings=[])

        valid_dtype = self.evaluate_dtype(series)
        if not valid_dtype:
            series = self.coerce(series)
            diagnostic["coerced"] = True
            valid_dtype = self.evaluate_dtype(series)

        if not valid_dtype:
            diagnostic["valid_dtype"] = False

        if self.check_nulls:
            nulls = self._evaluate_nulls(series)
            diagnostic["nulls"] = nulls

        if self.check_unique:
            uniqueness = self._evaluate_uniqueness(series)
            diagnostic["unique"] = uniqueness

        return series, diagnostic

    def evaluate_dtype(self, series: pd.Series) -> bool:
        return True

    def _evaluate_nulls(self, series: pd.Series, return_count=False) -> bool:
        nulls_count = series.isnull().sum()

        if return_count:
            return nulls_count > 0, nulls_count

        return nulls_count > 0

    def _evaluate_uniqueness(self, series: pd.Series) -> bool:
        return series.is_unique


class NumberColumn(ObjectColumn):
    def coerce(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='ignore')

    def evaluate_dtype(self, series: pd.Series) -> bool:
        return np.issubdtype(series.dtype, np.number)


class IntColumn(NumberColumn):

    def coerce(self, series: pd.Series) -> pd.Series:
        coerced = super().coerce(series)
        return coerced.astype(int)
    
    def evaluate_dtype(self, series: pd.Series) -> bool:
        return np.issubdtype(series.dtype, np.int_)

    def evaluate(self, series: pd.Series) -> Dict:
        series, diagnostic =  super().evaluate(series)
        diagnostic['warnings'].append('Null values on integer columns are not supported yet.')
        return series, diagnostic


class FloatColumn(NumberColumn):

    def coerce(self, series: pd.Series) -> pd.Series:
        coerced = super().coerce(series)
        return coerced.astype(float)
    
    def evaluate_dtype(self, series: pd.Series) -> bool:
        correct_dtype = []
        for prec in ('16', '32', '64'):
            correct_dtype.append(np.issubdtype(series.dtype, f'float{prec}'))
        return any(correct_dtype)


class StringColumn(ObjectColumn):
    
    def coerce(self, series: pd.Series) -> pd.Series:
        return series.astype(pd.StringDtype())

    def evaluate_dtype(self, series: pd.Series) -> bool:
        return 'string' == series.dtype(str)


class CategoryColumn(ObjectColumn):
    pass


class DatetimeColumn(ObjectColumn):
    pass


class TimedeltaColumn(ObjectColumn):
    pass
