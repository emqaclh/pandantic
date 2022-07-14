import pytest  # pylint: disable=unused-import

import pandas as pd

from pandantic import columns, shortcuts


def test_int_column_range_validated_correct():

    col = pd.Series([0, 0, 1])

    col_definition = columns.IntColumn(validations=[shortcuts.between_range(0, 5)])

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["pre_valid"]
    assert diagnostic["post_valid"]
    assert diagnostic["valid_dtype"]
    assert diagnostic["casted"] is False


def test_int_column_range_validated_wrong():

    col = pd.Series([0, 0, 1, 8])

    col_definition = columns.IntColumn(
        validations=[shortcuts.between_range(0, 5, mandatory=True)]
    )

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["pre_valid"]
    assert diagnostic["post_valid"] is False


def test_int_column_range_validated_wrong_pre():

    col = pd.Series([0, 0, 1, 8])

    col_definition = columns.IntColumn(
        validations=[
            shortcuts.between_range(0, 5, mandatory=True, requires_prevalidation=False)
        ]
    )

    _, diagnostic = col_definition.evaluate(col)

    assert diagnostic["pre_valid"] is False
    assert diagnostic["post_valid"] is None
