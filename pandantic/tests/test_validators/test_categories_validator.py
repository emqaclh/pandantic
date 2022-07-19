# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_categories_validator_correct_series():

    col = pd.Series(["a", "b", "c"])

    validator = validators.CategoriesValidator(
        mandatory=False, description="Data in range", categories=["a", "b", "c"]
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_categories_validator_wrong():

    col = pd.Series(["a", "b", "c", "d"])

    validator = validators.CategoriesValidator(
        mandatory=False, description="Data in range", categories=["a", "b", "c"]
    )

    series, validation = validator.evaluate(col)
    assert validation.original_issues == 1
    assert validation.valid is False
