# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy  as np

from src import validators

def test_unique_validator_correct_series():

    col = pd.Series(['a', 'b', 'c'])

    validator = validators.UniqueValidator(mandatory=False, description='Data in range')

    series, original_issue_count, issue_count, valid, amended = validator.evaluate(col)
    assert col.equals(series)
    assert not original_issue_count
    assert not issue_count
    assert valid
    assert amended is False

def test_unique_validator_wrong():

    col = pd.Series(['a', 'b', 'c', 'c', 'c', 'b'])

    validator = validators.UniqueValidator(mandatory=False, description='Data in range')

    _, original_issue_count, _, valid, _ = validator.evaluate(col)
    assert original_issue_count == 3
    assert valid is False