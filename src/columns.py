"""
Declares columns for main pandas datatypes:
Object, Numbers (float and int), Booleans, Datetime and Categories.
"""
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np

import abc

from src import validators


class Column(abc.ABC):

    check_nulls = False
    check_unique = False
    validations = None

    def __init__(
        self,
        check_nulls=False,
        check_unique=False,
        validations: Optional[Union[List, Tuple]] = None,
    ) -> None:
        self.check_nulls = check_nulls
        self.check_unique = check_unique
        if validations is not None:
            for validator in validations:
                if not isinstance(validator, validators.Validator):
                    raise ValueError(
                        "You must provide a list or tuple of validators only."
                    )
            self.validations = validations

    def _cast(self, column: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def evaluate(self, column: pd.Series) -> Tuple[pd.Series, Dict]:
        diagnostic = dict(casted=False, warnings=[])
        diagnostic = self._pre_evaluation(column, diagnostic=diagnostic)
        column, diagnostic = self._evaluate(column, diagnostic=diagnostic)
        diagnostic = self._post_evaluation(column, diagnostic)
        return column, diagnostic

    def _evaluate(self, column: pd.Series, diagnostic: Dict) -> Tuple[pd.Series, Dict]:

        if not isinstance(column, pd.Series):
            raise TypeError("A pandas.Series object must be provided.")

        valid_dtype = self._evaluate_dtype(column)
        if not valid_dtype:
            column = self._cast(column)
            diagnostic["casted"] = True
            valid_dtype = self._evaluate_dtype(column)

        diagnostic["valid_dtype"] = valid_dtype

        if self.check_nulls:
            nulls = self._evaluate_nulls(column)
            diagnostic["nulls"] = nulls

        if self.check_unique:
            uniqueness = self._evaluate_uniqueness(column)
            diagnostic["unique"] = uniqueness

        return column, diagnostic

    def _pre_evaluation(
        self, column: pd.Series, diagnostic: Dict  # pylint: disable=unused-argument
    ) -> Dict:
        return diagnostic

    def _post_evaluation(
        self, column: pd.Series, diagnostic: Dict  # pylint: disable=unused-argument
    ) -> Dict:
        return diagnostic

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        raise NotImplementedError()

    def _evaluate_nulls(self, column: pd.Series, return_count=False) -> bool:
        nulls_count = column.isnull().sum()

        if return_count:
            return nulls_count > 0, nulls_count

        return nulls_count > 0

    def _evaluate_uniqueness(self, column: pd.Series) -> bool:
        return column.is_unique


class ObjectColumn(Column):
    def _cast(self, column: pd.Series) -> pd.Series:
        return column

    def _evaluate_dtype(
        self, column: pd.Series  # pylint: disable=unused-argument
    ) -> bool:
        return True


class NumberColumn(ObjectColumn):
    def _cast(self, column: pd.Series) -> pd.Series:
        return pd.to_numeric(column, errors="ignore")

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        try:
            return np.issubdtype(column.dtype, np.number)
        except TypeError:
            return False


class IntColumn(NumberColumn):
    def __init__(self, check_nulls=False, check_unique=False) -> None:
        super().__init__(check_nulls, check_unique)
        self.check_nulls = True

    def _cast(self, column: pd.Series) -> pd.Series:
        try:
            coerced = super()._cast(column)
            return coerced.astype(int)
        except ValueError:
            return column

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        try:
            return np.issubdtype(column.dtype, np.int_)
        except TypeError:
            return False

    def _post_evaluation(self, column: pd.Series, diagnostic: Dict) -> Dict:
        diagnostic = super()._post_evaluation(column, diagnostic)

        if diagnostic["nulls"]:
            diagnostic["warnings"].append(
                "Null values on integer columns are not supported yet."
            )

        return diagnostic


class FloatColumn(NumberColumn):
    def _cast(self, column: pd.Series) -> pd.Series:
        try:
            coerced = super()._cast(column)
            return coerced.astype(float)
        except ValueError:
            return column

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        try:
            correct_dtype = []
            for prec in ("16", "32", "64"):
                correct_dtype.append(np.issubdtype(column.dtype, f"float{prec}"))
            return any(correct_dtype)
        except TypeError:
            return False


class StringColumn(ObjectColumn):
    def _cast(self, column: pd.Series) -> pd.Series:
        try:
            return column.astype(pd.StringDtype())
        except ValueError:
            return column

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return str(column.dtype) == "string"


class BoolColumn(ObjectColumn):
    def _cast(self, column: pd.Series) -> pd.Series:
        try:
            return column.astype(bool)
        except ValueError:
            return column

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return str(column.dtype) == "bool"


class CategoryColumn(ObjectColumn):

    __categories = None

    def __init__(
        self, check_nulls=False, check_unique=False, categories: Optional[List] = None
    ) -> None:
        super().__init__(check_nulls, check_unique)
        self.__categories = categories

    def _cast(self, column: pd.Series) -> pd.Series:
        return pd.Categorical(column, categories=self.__categories)

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return str(column.dtype) == "category"

    def _pre_evaluation(self, column: pd.Series, diagnostic: Dict) -> Dict:
        nunique_vals = column.nunique()
        nunique_cats = (
            len(self.__categories) if self.__categories is not None else nunique_vals
        )

        if nunique_cats != nunique_vals:
            diagnostic["warnings"].append(
                f"There are {nunique_vals} unique values on the column, but {nunique_cats} declared categories."
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

    def _cast(self, column: pd.Series) -> pd.Series:
        return pd.to_datetime(column, errors="ignore", format=self.__datetime_format)

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return "datetime" in str(column.dtype)
