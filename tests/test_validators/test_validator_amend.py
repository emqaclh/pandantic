# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import validators


def test_categories_validator_correct_series():

    col = pd.Series(["a", "b", "c", "d"])

    validator = validators.CategoriesValidator(
        mandatory=True, categories=["a", "b", "c"]
    ).add_amendment(lambda col: col.replace({"d": "c"}))

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert not col.equals(series)
    assert original_issue_count == 1
    assert issue_count == 0
    assert valid
    assert amended
