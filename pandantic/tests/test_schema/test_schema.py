# pylint: disable=unused-import
import pytest

import pandas as pd

from pandantic import columns, schemas


def test_schema_success():

    df = pd.DataFrame({"column_1": [0, 2, 3], "column_2": [True, True, False]})

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()

    schema_obj = TestSchema()

    _, evaluation = schema_obj.evaluate(df, "test")

    assert evaluation.column_1.valid
    assert evaluation.column_2.valid


def test_schema_wrong_invalid_dtype():

    df = pd.DataFrame(
        {
            "column_1": [0, 2, 3],
            "column_2": [True, True, False],
            "column_3": ["a", "b", "c"],
        }
    )

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()
        column_3 = columns.IntColumn()

    schema_obj = TestSchema()

    _, evaluation = schema_obj.evaluate(df, "test")

    assert evaluation.column_1.valid
    assert evaluation.column_2.valid
    assert evaluation.column_3.valid is False


def test_schema_wrong_castable():

    df = pd.DataFrame(
        {
            "column_1": [0, 2, 3],
            "column_2": [True, True, False],
            "column_3": ["1", "2", "2"],
        }
    )

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()
        column_3 = columns.IntColumn()

    schema_obj = TestSchema()

    _, evaluation = schema_obj.evaluate(df, "test")

    assert evaluation.column_1.valid
    assert not evaluation.column_1.amended
    assert evaluation.column_2.valid
    assert not evaluation.column_2.amended
    assert evaluation.column_3.valid
    assert evaluation.column_3.amended
