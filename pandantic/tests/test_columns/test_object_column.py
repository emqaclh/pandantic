# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import columns


def test_object_column_default_values():

    col = pd.Series([0, 0, 0]).astype(object)

    col_definition = columns.ObjectColumn()

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.amended is False
