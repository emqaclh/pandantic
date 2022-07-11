# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import columns


def test_datetime_column_correct_series():

    col = pd.Series([0, 0, 0])
    col = pd.to_datetime(col)

    col_definition = columns.DatetimeColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_datetime_column_wrong_series_castable():

    col = pd.Series([0, 0, 0])

    col_definition = columns.DatetimeColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]


def test_datetime_column_wrong_series_castable_by_format():

    col = pd.Series(['2020-01-01', '2020-01-01'])

    col_definition = columns.DatetimeColumn(datetime_format='%Y-%m-%d')

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]


def test_datetime_column_wrong_series_castable_wrong_format():

    col = pd.Series(['2020a-01-01', '2020-01-01'])

    col_definition = columns.DatetimeColumn(datetime_format='%Y-%m-%d')

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"] is False
    assert diagnostic["casted"]