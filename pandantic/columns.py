"""
Declares columns for main pandas datatypes:
Object, Numbers (float and int), Booleans, Datetime and Categories.
"""
import abc
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from pandantic import validators


class Column(abc.ABC):

    pre_validations: List[Type[validators.Validator]]
    post_validations: List[Type[validators.Validator]]

    def __init__(
        self,
        validations: Optional[Union[List, Tuple]] = None,
    ) -> None:
        self.pre_validations = []
        self.post_validations = []
        if validations is not None:
            for validator in validations:

                if not isinstance(validator, validators.Validator):
                    raise ValueError(
                        "You must provide a list or tuple of validators only."
                    )

                if validator.requires_prevalidation:
                    self.post_validations.append(validator)
                else:
                    self.pre_validations.append(validator)

            self.validations = validations

    def _cast(self, column: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def evaluate(self, column: pd.Series) -> Tuple[pd.Series, Dict]:
        diagnostic = dict(
            pre_validations=dict(),
            pre_valid=None,
            valid_dtype=None,
            casted=None,
            post_validations=dict(),
            post_valid=None,
        )
        column = column.copy()

        pre_valid, column, diagnostic = self._pre_validate(
            column, diagnostic=diagnostic
        )
        diagnostic["pre_valid"] = pre_valid

        if pre_valid:
            column, diagnostic = self._evaluate(column, diagnostic=diagnostic)

        if (
            pre_valid
            and diagnostic["valid_dtype"]
            and diagnostic["valid_dtype"] is not None
        ):

            post_valid, column, diagnostic = self._post_validate(
                column, diagnostic=diagnostic
            )
            diagnostic["post_valid"] = post_valid

        return column, diagnostic

    def _validate(self, column: pd.Series, diagnostic: Dict) -> Tuple[pd.Series, Dict]:
        if self.validations is None:
            return column, diagnostic

    def _pre_validate(
        self, column: pd.Series, diagnostic: Dict
    ) -> Tuple[bool, pd.Series, Dict]:
        column = column.copy()
        diagnostic["pre_validations"] = []
        able_to_continue = True
        for validator in self.pre_validations:
            if able_to_continue:

                (
                    column,
                    original_issue_count,
                    issue_count,
                    valid,
                    amended,
                ) = validator.evaluate(column)

                partial_diagnostic = dict(
                    original_issues=original_issue_count,
                    pending_issues=issue_count,
                    applied_amend=amended,
                    validated=valid,
                    description=validator.description,
                )

                diagnostic["pre_validations"].append(partial_diagnostic)

                if validator.mandatory and not valid:
                    able_to_continue = False
            else:

                partial_diagnostic = dict(
                    original_issues=None,
                    pending_issues=None,
                    applied_amend=None,
                    validated=None,
                    description=validator.description,
                )

                diagnostic["pre_validations"].append(partial_diagnostic)

        return able_to_continue, column, diagnostic

    def _post_validate(
        self, column: pd.Series, diagnostic: Dict
    ) -> Tuple[bool, pd.Series, Dict]:
        column = column.copy()
        diagnostic["post_validations"] = []
        able_to_continue = True
        for validator in self.post_validations:
            if able_to_continue:

                (
                    column,
                    original_issue_count,
                    issue_count,
                    valid,
                    amended,
                ) = validator.evaluate(column)

                partial_diagnostic = dict(
                    original_issues=original_issue_count,
                    pending_issues=issue_count,
                    applied_amend=amended,
                    validated=valid,
                    description=validator.description,
                )

                diagnostic["post_validations"].append(partial_diagnostic)

                if validator.mandatory and not valid:
                    able_to_continue = False
            else:

                partial_diagnostic = dict(
                    original_issues=None,
                    pending_issues=None,
                    applied_amend=None,
                    validated=None,
                    description=validator.description,
                )

                diagnostic["post_validations"].append(partial_diagnostic)

        return able_to_continue, column, diagnostic

    def _evaluate(self, column: pd.Series, diagnostic: Dict) -> Tuple[pd.Series, Dict]:

        if not isinstance(column, pd.Series):
            raise TypeError("A pandas.Series object must be provided.")

        valid_dtype = self._evaluate_dtype(column)
        if not valid_dtype:
            column = self._cast(column)
            diagnostic["casted"] = True
            valid_dtype = self._evaluate_dtype(column)
        else:
            diagnostic["casted"] = False

        diagnostic["valid_dtype"] = valid_dtype

        return column, diagnostic

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        raise NotImplementedError()


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
        self,
        validations: Optional[Union[List, Tuple]] = None,
        categories: Optional[List] = None,
    ) -> None:
        super().__init__(validations)
        self.__categories = categories

    def _cast(self, column: pd.Series) -> pd.Series:
        return pd.Categorical(column, categories=self.__categories)

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return str(column.dtype) == "category"


class DatetimeColumn(ObjectColumn):

    __datetime_format = None

    def __init__(
        self,
        datetime_format: Optional[str] = None,
        validations: Optional[Union[List, Tuple]] = None,
    ) -> None:
        super().__init__(validations)
        self.__datetime_format = datetime_format

    def _cast(self, column: pd.Series) -> pd.Series:
        return pd.to_datetime(column, errors="ignore", format=self.__datetime_format)

    def _evaluate_dtype(self, column: pd.Series) -> bool:
        return "datetime" in str(column.dtype)
