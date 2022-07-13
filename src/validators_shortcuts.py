"""
Shortcuts for validators instancing.
"""
from typing import Literal, List, Union, Pattern
from numbers import Number

import numpy as np

from src import validators


def between_range(
    min_value: Number,
    max_value: Number,
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    mandatory: bool = False,
    description: str = "N/A",
) -> validators.RangeValidator:
    return validators.RangeValidator(
        mandatory,
        description,
        min_value=min_value,
        max_value=max_value,
        inclusive=inclusive,
    )


def greater_than(
    min_value: Number, mandatory: bool = False, description: str = "N/A"
) -> validators.RangeValidator:
    return validators.RangeValidator(
        mandatory,
        description,
        min_value=min_value,
        max_value=np.inf,
        inclusive="neither",
    )


def greater_or_equal_than(
    min_value: Number, mandatory: bool = False, description: str = "N/A"
) -> validators.RangeValidator:
    return validators.RangeValidator(
        mandatory,
        description,
        min_value=min_value,
        max_value=np.inf,
        inclusive="left",
    )


def lower_than(
    max_value: Number, mandatory: bool = False, description: str = "N/A"
) -> validators.RangeValidator:
    return validators.RangeValidator(
        mandatory,
        description,
        min_value=-np.inf,
        max_value=max_value,
        inclusive="neither",
    )


def lower_or_equal_than(
    max_value: Number, mandatory: bool = False, description: str = "N/A"
) -> validators.RangeValidator:
    return validators.RangeValidator(
        mandatory,
        description,
        min_value=-np.inf,
        max_value=max_value,
        inclusive="right",
    )


def in_category(
    categories: List, mandatory: bool = False, description: str = "N/A"
) -> validators.CategoriesValidator:
    return validators.CategoriesValidator(mandatory, description, categories=categories)


def non_null(
    mandatory: bool = False, description: str = "N/A"
) -> validators.NonNullValidator:
    return validators.NonNullValidator(mandatory, description)


def is_unique(
    mandatory: bool = False, description: str = "N/A"
) -> validators.UniqueValidator:
    return validators.UniqueValidator(mandatory, description)


def match_pattern(
    pattern: Union[str, Pattern], mandatory: bool = False, description: str = "N/A"
) -> validators.PatternValidator:
    return validators.PatternValidator(mandatory, description, pattern=pattern)
