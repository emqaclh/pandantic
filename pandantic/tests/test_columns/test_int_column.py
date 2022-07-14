# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_int_column_correct_series():

    col = pd.Series([0, 0, 1])

    col_definition = columns.IntColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_int_column_wrong_series_float():

    col = pd.Series([0, 0, 1.5])

    col_definition = columns.IntColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
