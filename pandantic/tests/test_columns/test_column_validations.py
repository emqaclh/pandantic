import pytest  # pylint: disable=unused-import

import pandas as pd

from pandantic import columns, shortcuts


def test_int_column_range_validated_correct():

    col = pd.Series([0, 0, 1])

    col_definition = columns.IntColumn(
        column_validations=[shortcuts.between_range(0, 5)]
    )

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid
    assert evaluation.amended is False


def test_int_column_range_validated_wrong():

    col = pd.Series([0, 0, 1, 8])

    col_definition = columns.IntColumn(
        column_validations=[shortcuts.between_range(0, 5, mandatory=True)]
    )

    _, evaluation = col_definition.evaluate(col)

    assert evaluation.valid is False
