# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators


def test_unique_validator_correct_series():

    col = pd.Series(["a", "b", "c"])

    validator = validators.UniqueValidator(mandatory=False, description="Data in range")

    series, validation = validator.evaluate(col)
    assert col.equals(series)
    assert not validation.original_issues
    assert not validation.pending_issues
    assert validation.valid
    assert validation.amended is False


def test_unique_validator_wrong():

    col = pd.Series(["a", "b", "c", "c", "c", "b"])

    validator = validators.UniqueValidator(mandatory=False, description="Data in range")

    _, validation = validator.evaluate(col)
    assert validation.original_issues == 3
    assert validation.valid is False
