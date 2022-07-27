"""
Declares the base schema to evaluate and process pandas DataFrames.
"""
import abc
import warnings
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd
from collections import namedtuple

from pandantic import columns, evaluations, root_validator


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

        pre_root_validators, post_root_validators = self.get_model_root_validators()

        dataframe_evaluation = namedtuple(
            name,
            ["pre_root_evaluation", "post_root_validation"]
            + list(declared_columns.keys())
            + remaining_columns,
        )

        evaluation_data = dict()

        dataframe, pre_root_validation = pre_root_validators.validate(dataframe)
        evaluation_data["pre_root_evaluation"] = pre_root_validation

        pre_validations_status = [
            validation.valid
            for validation in pre_root_validation
            if validation.mandatory
        ]
        pre_valid = all(pre_validations_status)

        for column_name, column_declaration in declared_columns.items():
            if column_name not in missing_columns:
                if pre_valid:
                    column = dataframe.loc[:, column_name]
                    result_column, column_evaluation = column_declaration.evaluate(
                        column
                    )
                    dataframe.loc[:, column_name] = result_column
                else:
                    column_evaluation = evaluations.SuspendedColumnEvaluation()
            else:
                column_evaluation = evaluations.MissingColumn()
            evaluation_data[column_name] = column_evaluation

        for column_name in remaining_columns:
            evaluation_data[column_name] = evaluations.UnhandledColumn()

        dataframe, post_root_validation = post_root_validators.validate(dataframe)
        evaluation_data["post_root_validation"] = post_root_validation

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
            warnings.warn(
                f"""There is {len(missing_columns)} ({missing_columns}) missing columns,
                {len(remaining_columns)} ({remaining_columns}) remaining columns
                and {len(warning_columns)} ({warning_columns}) invalid non-mandatory evaluated columns.""",
                SchemaEvaluationWarning,
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

    @classmethod
    def get_model_root_validators(
        cls,
    ) -> Tuple[root_validator.RootValidatorSet, root_validator.RootValidatorSet]:
        class_attributes = cls.__dict__.items()
        pre_root_validators = root_validator.RootValidatorSet()
        post_root_validators = root_validator.RootValidatorSet()
        for name, value in class_attributes:
            if isinstance(value, classmethod):
                _callable = getattr(value, "__func__")
                is_root_validator = getattr(_callable, "root_validation", False)
                if is_root_validator:

                    main_func = _callable
                    pre = getattr(_callable, "pre", None)
                    description = getattr(_callable, "description", None)
                    if description is None:
                        description = name

                    mandatory = getattr(_callable, "mandatory", True)

                    validator = root_validator.RootValidator(
                        main_func=main_func,
                        mandatory=mandatory,
                        description=description,
                    )

                    amendment = getattr(_callable, "amendment", None)
                    if amendment is not None:
                        validator.set_amendment(amendment=amendment)

                    if pre:
                        pre_root_validators.add_validator(validator=validator)
                    else:
                        post_root_validators.add_validator(validator=validator)

        return pre_root_validators, post_root_validators


class SchemaEvaluationWarning(UserWarning):
    pass


class SchemaEvaluationException(Exception):
    evaluation: NamedTuple

    def __init__(self, *args: object, evaluation: NamedTuple) -> None:
        super().__init__(*args)
        self.evaluation = evaluation
