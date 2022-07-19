"""
Shortcuts for validators instancing.
"""
from typing import Literal, List, Union, Pattern, Optional
from numbers import Number

import numpy as np

from pandantic import validators


def between_range(
    min_value: Number,
    max_value: Number,
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    mandatory: Optional[bool] = None,
    description: Optional[str] = None,
) -> validators.RangeValidator:
    return validators.RangeValidator(
        min_value=min_value,
        max_value=max_value,
        inclusive=inclusive,
        mandatory=mandatory,
        description=description,
    )


def greater_than(
    min_value: Number, mandatory: bool = None, description: str = None
) -> validators.RangeValidator:
    return validators.RangeValidator(
        min_value=min_value,
        max_value=np.inf,
        inclusive="neither",
        mandatory=mandatory,
        description=description,
    )


def greater_or_equal_than(
    min_value: Number, mandatory: bool = None, description: str = None
) -> validators.RangeValidator:
    return validators.RangeValidator(
        min_value=min_value,
        max_value=np.inf,
        inclusive="left",
        mandatory=mandatory,
        description=description,
    )


def lower_than(
    max_value: Number, mandatory: bool = None, description: str = None
) -> validators.RangeValidator:
    return validators.RangeValidator(
        min_value=-np.inf,
        max_value=max_value,
        inclusive="neither",
        mandatory=mandatory,
        description=description,
    )


def lower_or_equal_than(
    max_value: Number, mandatory: bool = None, description: str = None
) -> validators.RangeValidator:
    return validators.RangeValidator(
        min_value=-np.inf,
        max_value=max_value,
        inclusive="right",
        mandatory=mandatory,
        description=description,
    )


def in_categories(
    categories: List, mandatory: bool = None, description: str = None
) -> validators.CategoriesValidator:
    return validators.CategoriesValidator(
        categories=categories, mandatory=mandatory, description=description
    )


def non_null(
    mandatory: bool = None, description: str = None
) -> validators.NonNullValidator:
    return validators.NonNullValidator(mandatory=mandatory, description=description)


def is_unique(
    mandatory: bool = None, description: str = None
) -> validators.UniqueValidator:
    return validators.UniqueValidator(mandatory=mandatory, description=description)


def match_pattern(
    pattern: Union[str, Pattern], mandatory: bool = None, description: str = None
) -> validators.PatternValidator:
    return validators.PatternValidator(
        pattern=pattern, mandatory=mandatory, description=description
    )
