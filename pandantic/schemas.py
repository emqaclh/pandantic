"""
Declares the base schema to evaluate and process pandas DataFrames.
"""
import abc
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd
from collections import namedtuple

from pandantic import columns, evaluations


class DataFrameModel(abc.ABC):
    def __init__(self) -> None:
        all_attributes = dir(self)
        column_attributes = [
            (attr, getattr(self, attr))
            for attr in all_attributes
            if isinstance(getattr(self, attr), columns.Column)
        ]
        column_attributes = dict(column_attributes)
        self.columns = column_attributes

    def evaluate(
        self, dataframe: pd.DataFrame, name: str
    ) -> Tuple[pd.DataFrame, NamedTuple]:
        if not name or name is None:
            raise ValueError("name should be correctly declared.")

        dataframe = dataframe.copy()

        original_column_names = list(dataframe.columns)
        dataframe.columns = self.transform_column_names(dataframe)

        declared_columns = self.get_columns()
        missing_columns, remaining_columns = self.check_columns(dataframe)

        dataframe_evaluation = namedtuple(
            name, list(declared_columns.keys()) + remaining_columns
        )

        evaluation_data = dict()
        for column_name, column_declaration in declared_columns.items():
            if column_name not in missing_columns:
                column = dataframe.loc[:, column_name]
                result_column, column_evaluation = column_declaration.evaluate(column)
                dataframe.loc[:, column_name] = result_column
            else:
                column_evaluation = evaluations.MissingColumn()
            evaluation_data[column_name] = column_evaluation

        for column_name in remaining_columns:
            evaluation_data[column_name] = evaluations.UnhandledColumn()

        dataframe.columns = original_column_names

        evaluation = dataframe_evaluation(**evaluation_data)

        all_valid = all(
            [
                _eval.valid
                for _eval in evaluation_data.values()
                if _eval.valid is not None
            ]
        )
        if not all_valid:
            raise SchemaEvaluationException(
                "There is invalid columns.", evaluation=evaluation
            )

        warning_columns = []
        for column_name, column_eval in evaluation_data.items():
            if column_eval.warnings and column_eval.warnings is not None:
                warning_columns.append(column_name)

        if missing_columns or remaining_columns or warning_columns:
            raise SchemaEvaluationWarning(
                f"There is {len(missing_columns)} missing columns, {len(remaining_columns)} remaining columns and {len(warning_columns)} invalid non-mandatory evaluated columns.",
                missing_columns=missing_columns,
                remaining_columns=remaining_columns,
                warning_columns=warning_columns,
            )

        return dataframe, evaluation

    def transform_column_names(self, dataframe: pd.DataFrame) -> List:
        return list(dataframe.columns)

    def check_columns(self, dataframe: pd.DataFrame) -> Tuple[List, List]:
        expected_cols = list(self.get_columns().keys())
        observed_cols = list(dataframe.columns)

        missing_cols = [col for col in expected_cols if col not in observed_cols]
        remaining_cols = [col for col in observed_cols if col not in expected_cols]

        return missing_cols, remaining_cols

    def get_columns(self) -> Dict[str, columns.Column]:
        return self.columns


class SchemaEvaluationWarning(UserWarning):
    missing_columns: List
    remaining_columns: List
    warning_columns: List

    def __init__(
        self,
        *args: object,
        missing_columns: List,
        remaining_columns: List,
        warning_columns: List,
    ) -> None:
        super().__init__(*args)
        if missing_columns is not None:
            self.missing_columns = missing_columns
        if remaining_columns is not None:
            self.remaining_columns = remaining_columns
        if warning_columns is not None:
            self.remaining_columns = warning_columns


class SchemaEvaluationException(Exception):
    evaluation: NamedTuple

    def __init__(self, *args: object, evaluation: NamedTuple) -> None:
        super().__init__(*args)
        self.evaluation = evaluation
