"""
Shortcuts for validators instancing.
"""
from typing import Literal, List
from numbers import Number

import numpy as np

from src import validators


def between_range(
    min_value: Number,
    max_value: Number,
    inclusive: Literal["both", "neither", "left", "right"] = "both",
):
    return validators.RangeValidator(
        min_value=min_value, max_value=max_value, inclusive=inclusive
    )


def greater_than(min_value: Number):
    return validators.RangeValidator(
        min_value=min_value,
        max_value=np.inf,
        inclusive="neither",
    )


def greater_or_equal_than(min_value: Number):
    return validators.RangeValidator(
        min_value=min_value,
        max_value=np.inf,
        inclusive="left",
    )


def lower_than(max_value: Number):
    return validators.RangeValidator(
        min_value=-np.inf,
        max_value=max_value,
        inclusive="neither",
    )


def lower_or_equal_than(max_value: Number):
    return validators.RangeValidator(
        min_value=-np.inf,
        max_value=max_value,
        inclusive="right",
    )


def in_category(categories: List):
    return validators.CategoriesValidator(categories=categories)
