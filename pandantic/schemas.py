"""
Declares the base schema to evaluate and process pandas DataFrames.
"""
import abc
from typing import Dict, List, Tuple

import pandas as pd

from pandantic import columns


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

    def evaluate(self, dataframe: pd.DataFrame) -> Tuple[bool, Dict, pd.DataFrame]:
        diagnostic = dict(columns=dict())
        original_column_names = list(dataframe.columns)
        dataframe.columns = self.transform_column_names(dataframe)

        declared_columns = self.get_columns()
        missing_columns, remaining_columns = self.check_columns(dataframe)

        for column_name, column_declaration in declared_columns.items():
            if column_name not in missing_columns:
                column = dataframe.loc[:, column_name]
                result_column, column_diagnostic = column_declaration.evaluate(column)
                dataframe.loc[:, column_name] = result_column
            else:
                column_diagnostic = dict(valid_dtype=False, extra="Missing column")
            diagnostic["columns"][column_name] = column_diagnostic

        diagnostic["missing_columns"] = missing_columns
        diagnostic["remaining_columns"] = remaining_columns

        dataframe.columns = original_column_names

        schema_eval = not missing_columns and not remaining_columns
        columns_eval = [
            bool(col["post_valid"]) for col in diagnostic["column"].values()
        ]

        general_eval = [schema_eval, all(columns_eval)]

        return all(general_eval), diagnostic, dataframe

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
