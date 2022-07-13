# pylint: disable=unused-import
import pytest

import pandas as pd

from src import columns, schemas


def test_schema_success():

    df = pd.DataFrame({"column_1": [0, 2, 3], "column_2": [True, True, False]})

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()

    schema_obj = TestSchema()

    _, diagnostic, _ = schema_obj.evaluate(df)

    assert diagnostic["columns"]["column_1"]["valid_dtype"]
    assert diagnostic["columns"]["column_2"]["valid_dtype"]


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

    _, diagnostic, _ = schema_obj.evaluate(df)

    assert diagnostic["columns"]["column_1"]["valid_dtype"]
    assert diagnostic["columns"]["column_2"]["valid_dtype"]
    assert not diagnostic["columns"]["column_3"]["valid_dtype"]


def test_schema_wrong_invalid_castable():

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

    _, diagnostic, _ = schema_obj.evaluate(df)

    assert diagnostic["columns"]["column_1"]["valid_dtype"]
    assert not diagnostic["columns"]["column_1"]["casted"]
    assert diagnostic["columns"]["column_2"]["valid_dtype"]
    assert not diagnostic["columns"]["column_2"]["casted"]
    assert diagnostic["columns"]["column_3"]["valid_dtype"]
    assert diagnostic["columns"]["column_3"]["casted"]
