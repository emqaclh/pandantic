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
        self.column_validators = validators.ValidatorSet()
        if column_validations is not None:
            for validator in column_validations:

                if not isinstance(validator, validators.Validator):
                    raise ValueError(
                        "You must provide a list or tuple of validators only."
                    )

                self.column_validators.add_validator(validator)

        any_datatype_validator = [
            isinstance(validator, datatype_validators.DatatypeValidator)
            for validator in self.column_validators
        ]
        if not any(any_datatype_validator) or column_validations is None:
            self.column_validators.add_validator(self.check_dtype())

        self.check_dtype_consistency()

    def check_dtype_consistency(self) -> None:
        inferred_dtype = self.infer_dtype()
        declared_dtype = type(self.check_dtype())
        if declared_dtype != inferred_dtype:
            raise UserWarning(
                f"Column declared dtype {str(declared_dtype)} differs from the one inferred from the validations {str(inferred_dtype)}."
            )

    def check_dtype(self) -> datatype_validators.DatatypeValidator:
        raise NotImplementedError()

    def infer_dtype(self) -> datatype_validators.DatatypeValidator:
        dtype_validators = [
            validator
            for validator in self.column_validators
            if isinstance(validator, datatype_validators.DatatypeValidator)
        ]
        last_dtype_validator = dtype_validators[-1]
        return type(last_dtype_validator)

    def evaluate(
        self, column: pd.Series
    ) -> Tuple[pd.Series, validations.ValidationSet]:

        column = column.copy()
        column, validation = self.column_validators.validate(column)
        column_eval = ColumnEvaluation(validation)
        return column, column_eval


class ColumnEvaluation:

    validation_set: validations.ValidationSet
    valid: bool
    amended: bool
    warnings: bool

    def __init__(self, validation_set: validations.ValidationSet) -> None:
        self.validation_set = validation_set
        self.check_validations()

    def check_validations(self):
        self.valid = all(
            [
                validation.valid
                for validation in self.validation_set
                if validation.mandatory
            ]
        )
        self.amended = any([validation.amended for validation in self.validation_set])
        self.warnings = any(
            [
                not validation.valid
                for validation in self.validation_set
                if not validation.mandatory
            ]
        )


class Column(BaseColumn):
    def check_dtype(self) -> datatype_validators.ObjectColumnValidator:
        return datatype_validators.ObjectColumnValidator()


class ObjectColumn(Column):
    pass


class NumberColumn(Column):
    def check_dtype(self) -> datatype_validators.NumericColumnValidator:
        return datatype_validators.NumericColumnValidator()


class IntColumn(Column):
    def check_dtype(self) -> datatype_validators.IntegerColumnValidator:
        return datatype_validators.IntegerColumnValidator()


class FloatColumn(Column):
    def check_dtype(self) -> datatype_validators.FloatColumnValidator:
        return datatype_validators.FloatColumnValidator()


class StringColumn(Column):
    def check_dtype(self) -> datatype_validators.StringColumnValidator:
        return datatype_validators.StringColumnValidator()


class BoolColumn(Column):
    def check_dtype(self) -> datatype_validators.BoolColumnValidator:
        return datatype_validators.BoolColumnValidator()


class CategoryColumn(Column):
    def check_dtype(self) -> datatype_validators.CategoryColumnValidator:
        return datatype_validators.CategoryColumnValidator()


class DatetimeColumn(Column):
    def __init__(
        self,
        column_validations: Optional[Union[List, Tuple]] = None,
        datetime_format: Optional[str] = None,
    ) -> None:
        self.datetime_format = datetime_format
        super().__init__(column_validations)

    def check_dtype(
        self, datetime_format: Optional[str] = None
    ) -> datatype_validators.DatetimeColumnValidator:
        _datetime_format = (
            datetime_format if datetime_format is not None else self.datetime_format
        )
        return datatype_validators.DatetimeColumnValidator(_datetime_format)
