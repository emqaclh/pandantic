# pylint: disable=unused-import
import pytest

import pandas as pd

from pandantic import columns, schemas
from pandantic.root_validator import root_validator


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

    with pytest.raises(schemas.SchemaEvaluationException) as error:

        schema_obj.evaluate(df, "test")

        assert error.evaluation.column_1.valid
        assert error.evaluation.column_2.valid
        assert error.evaluation.column_3.valid is False


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


def test_schema_warning_missing_cols():

    df = pd.DataFrame({"column_1": [0, 2, 3], "column_2": [True, True, False]})

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()
        column_3 = columns.IntColumn()

    schema_obj = TestSchema()

    with pytest.warns(schemas.SchemaEvaluationWarning) as record:
        _, evaluation = schema_obj.evaluate(df, "test")

        if not record:
            pytest.fail("Warning expected.")

        assert evaluation.column_1.valid


def test_schema_simple_root_pre_validator():

    df = pd.DataFrame({"column_1": [0, 2, 3], "column_2": [True, True, False]})

    class TestSchema(schemas.DataFrameModel):

        column_1 = columns.IntColumn()
        column_2 = columns.BoolColumn()

        @root_validator(
            pre=True,
            mandatory=True,
            amendment=lambda df: df.rename(columns={"column_3": "column_2"}),
        )
        def evaluate_column_right_name(cls, dataframe):
            return "column_3" in dataframe.columns

    schema_obj = TestSchema()

    _, evaluation = schema_obj.evaluate(df, "test")

    assert evaluation.pre_root_evaluation.valid
    assert evaluation.column_1.valid
    assert evaluation.column_2.valid
