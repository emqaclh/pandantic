"""
Declares columns for main pandas datatypes:
Object, Numbers (float and int), Booleans, Datetime and Categories.
"""
import abc
from typing import List, Optional, Tuple, Union

import pandas as pd

from pandantic import datatype_validators, validations, validators


class BaseColumn(abc.ABC):

    column_validators: validators.ValidatorSet

    def __init__(
        self,
        column_validations: Optional[Union[List, Tuple]] = None,
    ) -> None:
        column_validators = validators.ValidatorSet()
        if column_validations is not None:
            for validator in column_validations:

                if not isinstance(validator, validators.Validator):
                    raise ValueError(
                        "You must provide a list or tuple of validators only."
                    )

                column_validators.add_validator(validator)

        any_datatype_validator = [
            isinstance(validator, datatype_validators.DatatypeValidator)
            for validator in self.column_validators
        ]
        if not any(any_datatype_validator) or column_validations is None:
            self.column_validators.add_validator(self.check_dtype())

        self.check_dtype_consistency()

    def check_dtype_consistency(self) -> None:
        inferred_dtype = self.infer_dtype()
        if type(self.check_dtype) != inferred_dtype:
            raise UserWarning(
                "Column implicit dtype differs from the one inferred from the validations."
            )

    def check_dtype(self) -> datatype_validators.DatatypeValidator:
        raise NotImplementedError()

    def infer_dtype(self) -> datatype_validators.DatatypeValidator:
        dtype_validators = [
            validator
            for validator in self.column_validators.validators
            if isinstance(
                validator, datatype_validators, datatype_validators.DatatypeValidator
            )
        ]
        last_dtype_validator = dtype_validators[-1]
        return type(last_dtype_validator)

    def evaluate(
        self, column: pd.Series
    ) -> Tuple[pd.Series, validations.ValidationSet]:

        column = column.copy()
        column, validation = self.column_validators.validate(column)
        return column, validation


class Column(BaseColumn):
    def check_dtype(self) -> datatype_validators.ObjectColumnValidator:
        raise datatype_validators.ObjectColumnValidator()


class ObjectColumn(Column):
    pass


class NumberColumn(Column):
    def check_dtype(self) -> datatype_validators.NumericColumnValidator:
        raise datatype_validators.NumericColumnValidator()


class IntColumn(Column):
    def check_dtype(self) -> datatype_validators.IntegerColumnValidator:
        raise datatype_validators.IntegerColumnValidator()


class FloatColumn(Column):
    def check_dtype(self) -> datatype_validators.FloatColumnValidator:
        raise datatype_validators.FloatColumnValidator()


class StringColumn(Column):
    def check_dtype(self) -> datatype_validators.StringColumnValidator:
        raise datatype_validators.StringColumnValidator()


class BoolColumn(Column):
    def check_dtype(self) -> datatype_validators.BoolColumnValidator:
        raise datatype_validators.BoolColumnValidator()


class CategoryColumn(Column):
    def check_dtype(self) -> datatype_validators.CategoryColumnValidator:
        raise datatype_validators.CategoryColumnValidator()


class DatetimeColumn(Column):
    def check_dtype(
        self, datetime_format: Optional[str] = None
    ) -> datatype_validators.DatetimeColumnValidator:
        raise datatype_validators.DatetimeColumnValidator(datetime_format)
