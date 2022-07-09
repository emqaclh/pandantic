from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np


class ObjectColumn:

    check_nulls = False
    check_unique = False

    def __init__(self, check_nulls=False, check_unique=False) -> None:
        self.check_nulls = check_nulls
        self.check_unique = check_unique

    def cast(self, series: pd.Series) -> pd.Series:
        return series

    def evaluate(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        series, diagnostic = self._evaluate(series)
        diagnostic = self._check_for_warns(series, diagnostic)
        return series, diagnostic

    def _evaluate(self, series: pd.Series) -> Tuple[pd.Series, Dict]:

        if not isinstance(series, pd.Series):
            raise TypeError("A pandas.Series object must be provided")

        diagnostic = dict(casted=False)

        valid_dtype = self.evaluate_dtype(series)
        if not valid_dtype:
            series = self.cast(series)
            diagnostic["casted"] = True
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

    def _check_for_warns(self, series: pd.Series, diagnostic: Dict) -> Dict:
        diagnostic["warnings"] = []
        return diagnostic

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
    def cast(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="ignore")

    def evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            return np.issubdtype(series.dtype, np.number)
        except TypeError:
            return False


class IntColumn(NumberColumn):
    def cast(self, series: pd.Series) -> pd.Series:
        try:
            coerced = super().cast(series)
            return coerced.astype(int)
        except ValueError:
            return series

    def evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            return np.issubdtype(series.dtype, np.int_)
        except TypeError:
            return False

    def _check_for_warns(self, series: pd.Series, diagnostic: Dict) -> Dict:
        diagnostic = super()._check_for_warns(series, diagnostic)

        if diagnostic["nulls"]:
            diagnostic["warnings"].append(
                "Null values on integer columns are not supported yet."
            )

        return diagnostic


class FloatColumn(NumberColumn):
    def cast(self, series: pd.Series) -> pd.Series:
        try:
            coerced = super().cast(series)
            return coerced.astype(float)
        except ValueError:
            return series

    def evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            correct_dtype = []
            for prec in ("16", "32", "64"):
                correct_dtype.append(np.issubdtype(series.dtype, f"float{prec}"))
            return any(correct_dtype)
        except TypeError:
            return False


class StringColumn(ObjectColumn):
    def cast(self, series: pd.Series) -> pd.Series:
        try:
            return series.astype(pd.StringDtype())
        except ValueError:
            return series

    def evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "string"


class BoolColumns(ObjectColumn):
    def cast(self, series: pd.Series) -> pd.Series:
        try:
            return series.astype(bool)
        except ValueError:
            return series

    def evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "bool"


class CategoryColumn(ObjectColumn):
    
    def cast(self, series: pd.Series) -> pd.Series:
        return pd.Categorical(series)
    
    def evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "str"


class DatetimeColumn(ObjectColumn):

    __datetime_format = None

    def __init__(self, check_nulls=False, check_unique=False, datetime_format: Optional[str]=None) -> None:
        super().__init__(check_nulls, check_unique)
        self.__datetime_format = datetime_format
    
    def cast(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors='ignore', format=self.__datetime_format)
    
    def evaluate_dtype(self, series: pd.Series) -> bool:
        return 'datetime' in str(series.dtype)
