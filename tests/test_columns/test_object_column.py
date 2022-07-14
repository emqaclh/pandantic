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
