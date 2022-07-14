# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_string_column_correct_series():

    col = pd.Series(["a", "b", "c"]).astype(pd.StringDtype())

    col_definition = columns.StringColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_string_column_wrong_series_str_like_object():

    col = pd.Series(["a", "b", "c"])

    col_definition = columns.StringColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]


def test_string_column_wrong_series_int():

    col = pd.Series([1, 2, 3])

    col_definition = columns.StringColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]


def test_string_column_wrong_series_bool():

    col = pd.Series([True])

    col_definition = columns.StringColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
