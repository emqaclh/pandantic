# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import columns


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


def test_category_column_wrong_series_missing_cats():

    col = pd.Series(["a", "b", "c"])

    col_definition = columns.CategoryColumn(categories=["a", "b"])

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
    assert (
        "There are 3 unique values on the column, but 2 declared categories."
        in diagnostic["warnings"]
    )


def test_category_column_wrong_series_additional_cats():

    col = pd.Series(["a", "b"])

    col_definition = columns.CategoryColumn(categories=["a", "b", "c"])

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"]
    assert (
        "There are 2 unique values on the column, but 3 declared categories."
        in diagnostic["warnings"]
    )
