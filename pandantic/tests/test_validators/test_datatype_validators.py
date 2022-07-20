# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import datatype_validators


def test_numeric_dtype_validator_correct_series():

    col = pd.Series([1, 0.3, 0.5])

    validator = datatype_validators.NumericColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_numeric_dtype_validator_amendable_series():

    col = pd.Series(["1", ".3", "5"])

    validator = datatype_validators.NumericColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended


def test_integer_dtype_validator_correct_series():

    col = pd.Series([1, 3, 5])

    validator = datatype_validators.IntegerColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_integer_dtype_validator_amendable_series():

    col = pd.Series(["1", "3", "5"])

    validator = datatype_validators.IntegerColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_float_dtype_validator_correct_series():

    col = pd.Series([0.1, 3, 5])

    validator = datatype_validators.FloatColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_float_dtype_validator_amendable_series():

    col = pd.Series(["1", ".3", "5", np.nan])

    validator = datatype_validators.FloatColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_string_dtype_validator_correct_series():

    col = pd.Series(["1", ".3", "5"]).astype(pd.StringDtype())

    validator = datatype_validators.StringColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_string_dtype_validator_amendable_series():

    col = pd.Series([0.1, 3, 5])

    validator = datatype_validators.StringColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_bool_dtype_validator_correct_series():

    col = pd.Series([True, False, True]).astype(bool)

    validator = datatype_validators.BoolColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_bool_dtype_validator_amendable_series():

    col = pd.Series([0, 1, 1])

    validator = datatype_validators.BoolColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_categorical_dtype_validator_correct_series():

    col = pd.Categorical(pd.Series(["a", "b", "b"]))

    validator = datatype_validators.CategoryColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_categorical_dtype_validator_amendable_series():

    col = pd.Series(["a", "b", "b"])

    validator = datatype_validators.CategoryColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_datetime_dtype_validator_correct_series():

    col = pd.Series([0, 0, 0])
    col = pd.to_datetime(col)

    validator = datatype_validators.DatetimeColumnValidator()

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_datetime_dtype_validator_amendable_series():

    col = pd.Series([0, 0, 0])

    validator = datatype_validators.DatetimeColumnValidator()

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid


def test_datetime_dtype_validator_amendable_series_format():

    col = pd.Series(["2020-01-01", "2020-01-01"])

    validator = datatype_validators.DatetimeColumnValidator(datetime_format="%Y-%m-%d")

    _, validation = validator.evaluate(col)
    assert validation.original_issues
    assert validation.amended
    assert not validation.pending_issues
    assert validation.valid
