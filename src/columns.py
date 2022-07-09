"""
Declares columns for main pandas datatypes.
"""
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


class ObjectColumn:

    check_nulls = False
    check_unique = False

    def __init__(self, check_nulls=False, check_unique=False) -> None:
        self.check_nulls = check_nulls
        self.check_unique = check_unique

    def _cast(self, series: pd.Series) -> pd.Series:
        return series

    def evaluate(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        diagnostic = dict(casted=False, warnings=[])
        diagnostic = self._pre_evaluation(series, diagnostic=diagnostic)
        series, diagnostic = self._evaluate(series, diagnostic=diagnostic)
        diagnostic = self._post_evaluation(series, diagnostic)
        return series, diagnostic

    def _evaluate(self, series: pd.Series, diagnostic: Dict) -> Tuple[pd.Series, Dict]:

        if not isinstance(series, pd.Series):
            raise TypeError("A pandas.Series object must be provided")

        valid_dtype = self._evaluate_dtype(series)
        if not valid_dtype:
            series = self._cast(series)
            diagnostic["casted"] = True
            valid_dtype = self._evaluate_dtype(series)

        if not valid_dtype:
            diagnostic["valid_dtype"] = False

        if self.check_nulls:
            nulls = self._evaluate_nulls(series)
            diagnostic["nulls"] = nulls

        if self.check_unique:
            uniqueness = self._evaluate_uniqueness(series)
            diagnostic["unique"] = uniqueness

        return series, diagnostic

    # pylint: disable=unused-argument
    def _pre_evaluation(self, series: pd.Series, diagnostic: Dict) -> Dict:
        return diagnostic

    # pylint: disable=unused-argument
    def _post_evaluation(self, series: pd.Series, diagnostic: Dict) -> Dict:
        return diagnostic

    # pylint: disable=unused-argument
    def _evaluate_dtype(self, series: pd.Series) -> bool:
        return True

    def _evaluate_nulls(self, series: pd.Series, return_count=False) -> bool:
        nulls_count = series.isnull().sum()

        if return_count:
            return nulls_count > 0, nulls_count

        return nulls_count > 0

    def _evaluate_uniqueness(self, series: pd.Series) -> bool:
        return series.is_unique


class NumberColumn(ObjectColumn):
    def _cast(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="ignore")

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            return np.issubdtype(series.dtype, np.number)
        except TypeError:
            return False


class IntColumn(NumberColumn):
    def _cast(self, series: pd.Series) -> pd.Series:
        try:
            coerced = super()._cast(series)
            return coerced.astype(int)
        except ValueError:
            return series

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            return np.issubdtype(series.dtype, np.int_)
        except TypeError:
            return False

    def _post_evaluation(self, series: pd.Series, diagnostic: Dict) -> Dict:
        diagnostic = super()._post_evaluation(series, diagnostic)

        if diagnostic["nulls"]:
            diagnostic["warnings"].append(
                "Null values on integer columns are not supported yet."
            )

        return diagnostic


class FloatColumn(NumberColumn):
    def _cast(self, series: pd.Series) -> pd.Series:
        try:
            coerced = super()._cast(series)
            return coerced.astype(float)
        except ValueError:
            return series

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        try:
            correct_dtype = []
            for prec in ("16", "32", "64"):
                correct_dtype.append(np.issubdtype(series.dtype, f"float{prec}"))
            return any(correct_dtype)
        except TypeError:
            return False


class StringColumn(ObjectColumn):
    def _cast(self, series: pd.Series) -> pd.Series:
        try:
            return series.astype(pd.StringDtype())
        except ValueError:
            return series

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "string"


class BoolColumns(ObjectColumn):
    def _cast(self, series: pd.Series) -> pd.Series:
        try:
            return series.astype(bool)
        except ValueError:
            return series

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "bool"


class CategoryColumn(ObjectColumn):

    __categories = None

    def __init__(
        self, check_nulls=False, check_unique=False, categories: Optional[List] = None
    ) -> None:
        super().__init__(check_nulls, check_unique)
        self.__categories = categories

    def _cast(self, series: pd.Series) -> pd.Series:
        return pd.Categorical(series, categories=self.__categories)

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        return str(series) == "str"

    def _pre_evaluation(self, series: pd.Series, diagnostic: Dict) -> Dict:
        nunique_vals = series.nunique()
        nunique_cats = (
            len(self.__categories) if self.__categories is not None else nunique_vals
        )

        if nunique_cats != nunique_vals:
            diagnostic["warnings"].append(
                f"There is {nunique_vals} unique values on the column, but {nunique_cats} declared categories."
            )

        return diagnostic


class DatetimeColumn(ObjectColumn):

    __datetime_format = None

    def __init__(
        self,
        check_nulls=False,
        check_unique=False,
        datetime_format: Optional[str] = None,
    ) -> None:
        super().__init__(check_nulls, check_unique)
        self.__datetime_format = datetime_format

    def _cast(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="ignore", format=self.__datetime_format)

    def _evaluate_dtype(self, series: pd.Series) -> bool:
        return "datetime" in str(series.dtype)
