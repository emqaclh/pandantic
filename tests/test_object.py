# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from src import columns


def test_object_column_default_values():

    col = pd.Series([0, 0, 0])

    col_definition = columns.ObjectColumn()

    _, diagnostic = col_definition.evaluate(col)

    assert "nulls" not in diagnostic
    assert "unique" not in diagnostic
    assert diagnostic["casted"] is False


def test_object_column_uniqueness():

    col = pd.Series([0, 1, 2])

    col_definition = columns.ObjectColumn(check_unique=True)

    _, diagnostic = col_definition.evaluate(col)

    assert "unique" in diagnostic
    assert diagnostic["unique"] is True


def test_object_column_nulls():

    col = pd.Series([0, 1, 2, np.nan])

    col_definition = columns.ObjectColumn(check_nulls=True)

    _, diagnostic = col_definition.evaluate(col)

    assert "nulls" in diagnostic
    assert diagnostic["nulls"]
