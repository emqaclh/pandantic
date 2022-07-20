# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_number_column_correct_series():

    col = pd.Series([0, 0, 1, 2.5, np.nan])

    col_definition = columns.NumberColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended is False


def test_number_column_wrong_series():

    col = pd.Series([0, 0, 1, 2.5, np.nan, "a"])

    col_definition = columns.NumberColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid is False
    assert evaluation.amended


def test_number_column_fixable_series():

    col = pd.Series([0, 0, 1, 2.5, np.nan, "1"])

    col_definition = columns.NumberColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended
