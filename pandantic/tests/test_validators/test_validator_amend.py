# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_categories_validator_correct_series():

    col = pd.Series(["a", "b", "c", "d"])

    validator = validators.CategoriesValidator(
        mandatory=True, categories=["a", "b", "c"]
    ).set_amendment(lambda col: col.replace({"d": "c"}))

    series, validation = validator.evaluate(col)
    assert not col.equals(series)
    assert validation.original_issues == 1
    assert validation.pending_issues == 0
    assert validation.valid
    assert validation.amended
