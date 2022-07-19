"""
Specific validators for datatype validation.
"""
import abc
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pandantic import validators


class DatatypeValidator(validators.Validator, abc.ABC):
    pass


class NumericColumnValidator(DatatypeValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is a numeric dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: pd.to_numeric(column, errors="ignore")

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:

        valid_dtype = np.issubdtype(column.dtype, np.number)

        return 0 if valid_dtype else len(column), valid_dtype


class IntegerColumnValidator(NumericColumnValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is integer numeric dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: pd.to_numeric(
            column, downcast="integer", errors="ignore"
        )

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        valid_dtype = np.issubdtype(column.dtype, np.int_)

        return 0 if valid_dtype else len(column), valid_dtype


class FloatColumnValidator(DatatypeValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is float numeric dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: pd.to_numeric(
            column, downcast="float", errors="ignore"
        )

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:

        correct_dtype = []
        for prec in ("16", "32", "64"):
            correct_dtype.append(np.issubdtype(column.dtype, f"float{prec}"))

        valid_dtype = any(correct_dtype)

        return 0 if valid_dtype else len(column), valid_dtype


class StringColumnValidator(DatatypeValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is string dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: column.astype(pd.StringDtype())

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        valid_dtype = str(column.dtype) == "string"

        return 0 if valid_dtype else len(column), valid_dtype


class BoolColumnValidator(DatatypeValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is bool dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: column.astype(bool)

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        valid_dtype = str(column.dtype) == "bool"

        return 0 if valid_dtype else len(column), valid_dtype


class CategoryColumnValidator(DatatypeValidator):
    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
    ) -> None:

        if description is None:
            description = "Column is categorical dtype."

        super().__init__(mandatory, description, requires_prevalidation)

        self.amendment = lambda column: pd.Categorical(
            column
        )  # pylint: disable=unnecessary-lambda

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        valid_dtype = str(column.dtype) == "category"

        return 0 if valid_dtype else len(column), valid_dtype


class DatetimeColumnValidator(DatatypeValidator):

    __datetime_format = None

    def __init__(
        self,
        mandatory: bool = True,
        description: str = None,
        requires_prevalidation: bool = True,
        datetime_format: Optional[str] = None,
    ) -> None:

        if description is None:
            description = "Column is a datetime dtype."

        super().__init__(mandatory, description, requires_prevalidation)
        self.__datetime_format = datetime_format

        self.amendment = lambda column: pd.to_datetime(
            column, errors="ignore", format=self.__datetime_format
        )

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        valid_dtype = "datetime" in str(column.dtype)

        return 0 if valid_dtype else len(column), valid_dtype
