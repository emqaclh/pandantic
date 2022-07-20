# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_datetime_column_correct_series():

    col = pd.Series([0, 0, 0])
    col = pd.to_datetime(col)

    col_definition = columns.DatetimeColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended is False


def test_datetime_column_wrong_series_castable():

    col = pd.Series([0, 0, 0])

    col_definition = columns.DatetimeColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended


def test_datetime_column_wrong_series_castable_by_format():

    col = pd.Series(["2020-01-01", "2020-01-01"])

    col_definition = columns.DatetimeColumn(datetime_format="%Y-%m-%d")

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended


def test_datetime_column_wrong_series_castable_wrong_format():

    col = pd.Series(["2020a-01-01", "2020-01-01"])

    col_definition = columns.DatetimeColumn(datetime_format="%Y-%m-%d")

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid is False
    assert evaluation.amended
