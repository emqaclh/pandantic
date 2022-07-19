# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_range_validator_correct_series():

    col = pd.Series([1, 2, 3])

    validator = validators.RangeValidator(
        mandatory=False, description="Data in range", min_value=0, max_value=5
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_range_validator_wrong_series():

    col = pd.Series([1, 2, 3, 8])

    validator = validators.RangeValidator(
        mandatory=False, description="Data in range", min_value=0, max_value=5
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert validation.original_issues
    assert validation.pending_issues
    assert validation.valid is False
    assert validation.amended is False


def test_range_validator_wrong_series_boundaries():

    col = pd.Series([1, 2, 3, 8])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=8,
        inclusive="left",
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert validation.original_issues
    assert validation.pending_issues
    assert validation.valid is False
    assert validation.amended is False

    col = pd.Series([0, 2, 3, 7])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=8,
        inclusive="neither",
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert validation.original_issues
    assert validation.pending_issues
    assert validation.valid is False
    assert validation.amended is False


def test_range_validator_lt():

    col = pd.Series([1, 2, 3, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=-np.inf,
        max_value=15,
        inclusive="neither",
    )

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert validation.original_issues == 1
    assert validation.valid is False


def test_range_validator_le():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=-np.inf,
        max_value=15,
        inclusive="right",
    )

    _, validation = validator.evaluate(col)
    assert not validation.original_issues
    assert validation.valid


def test_range_validator_gt():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=np.inf,
        inclusive="left",
    )

    _, validation = validator.evaluate(col)
    assert not validation.original_issues
    assert validation.valid


def test_range_validator_ge():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=1,
        max_value=np.inf,
        inclusive="left",
    )

    _, validation = validator.evaluate(col)
    assert not validation.original_issues
    assert validation.valid
