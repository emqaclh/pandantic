# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import validators


def test_categories_validator_correct_series():

    col = pd.Series(["a", "b", "c"])

    validator = validators.CategoriesValidator(
        mandatory=False, description="Data in range", categories=["a", "b", "c"]
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert not original_issue_count
    assert not issue_count
    assert valid
    assert amended is False


def test_categories_validator_wrong():

    col = pd.Series(["a", "b", "c", "d"])

    validator = validators.CategoriesValidator(
        mandatory=False, description="Data in range", categories=["a", "b", "c"]
    )

    _, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert original_issue_count == 1
    assert valid is False
