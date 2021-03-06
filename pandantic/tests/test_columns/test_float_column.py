# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_float_column_correct_series():

    col = pd.Series([0.1, 0.2, 0.4])

    col_definition = columns.FloatColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended is False


def test_float_column_wrong_series_int():

    col = pd.Series([0, 0, 1])

    col_definition = columns.FloatColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended


def test_float_column_wrong_series_string():

    col = pd.Series([0, 0, 1, "a"])

    col_definition = columns.FloatColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid is False
    assert evaluation.amended
