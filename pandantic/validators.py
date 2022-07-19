"""
Validators for data validation and amendment.
"""
import abc
from numbers import Number
from typing import Callable, List, Literal, Pattern, Tuple, Type, Union

import numpy as np
import pandas as pd

from pandantic import validations


class Validator(abc.ABC):
    def __init__(self, mandatory: bool = True, description: str = None) -> None:
        self.mandatory = mandatory if mandatory is not None else False
        self.description = description if description is not None else "N/A"
        self.amendment = None

    def evaluate(self, column) -> Tuple[pd.Series, validations.Validation]:
        if not isinstance(column, pd.Series):
            raise TypeError("A pandas.Series object must be provided")

        column = column.copy()

        try:

            validation = validations.Validation(self.description)

            original_issue_count, valid = self._evaluate(column)
            validation.original_issues = original_issue_count
            validation.pending_issues = original_issue_count

            if not valid and self.amendment is not None:
                column = self.amendment(column)
                issue_count, valid = self._evaluate(column)
                validation.pending_issues = issue_count
                validation.amended = True

            validation.valid = valid

            return column, validation

        except Exception as error:
            raise validations.ValidationError(self.description, error).with_traceback(
                error.__traceback__
            )

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        raise NotImplementedError()

    def add_amendment(
        self, amendment: Callable[[pd.Series], pd.Series]
    ) -> Type["Validator"]:
        self.amendment = amendment
        return self


class ValidatorSet:

    validators: List[Validator]

    def __init__(self) -> None:
        self.validators = []

    def add_validator(self, validator: Validator):
        if not isinstance(validator, Validator):
            raise ValueError(f"Validator expected, got {type(validator)} instead.")
        else:
            self.validators.append(validator)

    def validate(
        self, column: pd.Series
    ) -> Tuple[pd.Series, validations.ValidationSet]:

        column = column.copy()
        validation_set = validations.ValidationSet()
        keep_validating = True

        for validator in self.validators:

            if keep_validating:
                try:
                    column, validation = validator.evaluate(column)
                except validations.ValidationError as error:
                    validation = error
            else:
                validation = validations.SuspendedValidation(validator.description)

            validation_set.add_validation(validation)
            if (
                (validation.valid is False and validator.mandatory)
                or isinstance(validation, validations.ValidationError)
                or (not keep_validating)
            ):
                keep_validating = False

        return column, validation_set


class RangeValidator(Validator):
    def __init__(
        self,
        min_value: Number,
        max_value: Number,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
        mandatory: bool = True,
        description: str = None,
    ) -> None:

        if min_value is None or max_value is None:
            raise ValueError("min_value and max_value must be provided.")

        if description is None:
            description = f"Values are between {min_value} and {max_value} ({inclusive} inclusive)"

        super().__init__(mandatory, description)

        self.inclusive = inclusive
        self.min_value, self.max_value = min_value, max_value

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        result = False
        non_null = column.count()

        if np.isinf(self.max_value):
            if self.inclusive == "left" or self.inclusive == "both":
                result = column.ge(self.min_value)
            else:
                result = column.gt(self.min_value)

        elif np.isneginf(self.min_value):
            if self.inclusive == "right" or self.inclusive == "both":
                result = column.le(self.max_value)
            else:
                result = column.lt(self.max_value)

        else:
            result = column.between(
                self.min_value, self.max_value, inclusive=self.inclusive
            )

        result = result.sum()

        return (non_null - result), not ((non_null - result) > 0)


class CategoriesValidator(Validator):
    def __init__(
        self, categories: List, mandatory: bool = True, description: str = None
    ) -> None:

        if not len(categories):
            raise ValueError("Categories list cannot be empty.")

        if description is None:
            description = f'Possible values: {", ".join(categories) if len(categories) < 7 else ", ".join(categories[:3]) + " … " + ", ".join(categories[:-3])}.'

        super().__init__(mandatory, description)

        self.categories = categories

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:
        non_null = column.count()

        in_category = column.isin(self.categories).sum()

        return (non_null - in_category), not ((non_null - in_category) > 0)


class NonNullValidator(Validator):
    def __init__(self, mandatory: bool = True, description: str = None) -> None:

        if description is None:
            description = "No null values."

        super().__init__(mandatory, description)

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:

        null_values = column.isnull().sum()

        return null_values, not (null_values)


class UniqueValidator(Validator):
    def __init__(self, mandatory: bool = True, description: str = None) -> None:

        if description is None:
            description = "Only unique values."

        super().__init__(mandatory, description)

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:

        non_unique = column.duplicated(keep="first").sum()

        return non_unique, not (non_unique)


class PatternValidator(Validator):
    def __init__(
        self,
        pattern: Union[str, Pattern],
        mandatory: bool = True,
        description: str = None,
    ) -> None:

        if description is None:
            description = f"Values matches {pattern}."

        super().__init__(mandatory, description)
        self.pattern = pattern

    def _evaluate(self, column: pd.Series) -> Tuple[int, bool]:

        non_null = column.count()
        match_count = column.str.fullmatch(self.pattern, case=True).sum()

        return (non_null - match_count), not (non_null - match_count) > 0
