# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_pattern_validator_correct_series():

    col = pd.Series(["a_", "b_", "c_"])

    validator = validators.PatternValidator(
        mandatory=False, description="Data in range", pattern=r".{1}_"
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.original_issues
    assert validation.valid
    assert validation.amended is False


def test_pattern_validator_wrong_series():

    col = pd.Series(["_a_", "b_", "c_"])

    validator = validators.PatternValidator(
        mandatory=False, description="Data in range", pattern=r".{1}_"
    )

    _, validation = validator.evaluate(col)
    assert validation.original_issues == 1
