# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_category_column_correct_series():

    col = pd.Series(["a", "b", "c"]).astype("category")

    col_definition = columns.CategoryColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_category_column_wrong_series_castable():

    col = pd.Series(["a", "b", "c"])

    col_definition = columns.CategoryColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
