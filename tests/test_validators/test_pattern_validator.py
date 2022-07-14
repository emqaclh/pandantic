# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import validators


def test_pattern_validator_correct_series():

    col = pd.Series(["a_", "b_", "c_"])

    validator = validators.PatternValidator(
        mandatory=False, description="Data in range", pattern=r".{1}_"
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert not original_issue_count
    assert not issue_count
    assert valid
    assert amended is False


def test_pattern_validator_wrong_series():

    col = pd.Series(["_a_", "b_", "c_"])

    validator = validators.PatternValidator(
        mandatory=False, description="Data in range", pattern=r".{1}_"
    )

    _, original_issue_count, _, _, _ = validator.evaluate(col)
    assert original_issue_count == 1
