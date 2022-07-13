# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import columns


def test_bool_column_correct_series():

    col = pd.Series([True, True, False])

    col_definition = columns.BoolColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_bool_column_wrong_series_misc_values():

    col = pd.Series([True, 1, 2.4, 0, 8, "a"])

    col_definition = columns.BoolColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
