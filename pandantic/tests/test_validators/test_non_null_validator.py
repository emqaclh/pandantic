# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_non_null_validator_correct_series():

    col = pd.Series(["a", "b", "c"])

    validator = validators.NonNullValidator(
        mandatory=False, description="Data in range"
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_non_null_validator_wrong():

    col = pd.Series([0, 1, np.nan, 5, np.nan])

    validator = validators.NonNullValidator(
        mandatory=False, description="Data in range"
    )

    _, validation = validator.evaluate(col)
    assert validation.original_issues == 2
    assert validation.valid is False
