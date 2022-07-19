# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import validators


def test_range_validator_correct_series():

    col = pd.Series([1, 2, 3])

    validator = validators.RangeValidator(
        mandatory=False, description="Data in range", min_value=0, max_value=5
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert not original_issue_count
    assert not issue_count
    assert valid
    assert amended is False


def test_range_validator_wrong_series():

    col = pd.Series([1, 2, 3, 8])

    validator = validators.RangeValidator(
        mandatory=False, description="Data in range", min_value=0, max_value=5
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert original_issue_count
    assert issue_count
    assert valid is False
    assert amended is False


def test_range_validator_wrong_series_boundaries():

    col = pd.Series([1, 2, 3, 8])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=8,
        inclusive="left",
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert original_issue_count
    assert issue_count
    assert valid is False
    assert amended is False

    col = pd.Series([0, 2, 3, 7])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=8,
        inclusive="neither",
    )

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert original_issue_count
    assert issue_count
    assert valid is False
    assert amended is False


def test_range_validator_lt():

    col = pd.Series([1, 2, 3, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=-np.inf,
        max_value=15,
        inclusive="neither",
    )

    series, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert col.equals(series)
    assert original_issue_count == 1
    assert valid is False


def test_range_validator_le():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=-np.inf,
        max_value=15,
        inclusive="right",
    )

    _, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert not original_issue_count
    assert valid


def test_range_validator_gt():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=0,
        max_value=np.inf,
        inclusive="left",
    )

    _, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert not original_issue_count
    assert valid


def test_range_validator_ge():

    col = pd.Series([1, 2, 3, 8, 15])

    validator = validators.RangeValidator(
        mandatory=False,
        description="Data in range",
        min_value=1,
        max_value=np.inf,
        inclusive="left",
    )

    _, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert not original_issue_count
    assert valid
