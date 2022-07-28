from typing import Callable, T, Optional, Type, Tuple

import pandas as pd

from pandantic import validators, validations


def root_validator(
    pre=True,
    mandatory=True,
    description: Optional[str] = None,
    amendment: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    def _decorator(func: Callable[[T, pd.DataFrame], bool]):
        func.root_validation = True
        func.pre = pre
        func.amendment = amendment
        func.mandatory = mandatory
        func.description = description
        func = classmethod(func)
        return func

    return _decorator


class RootValidator(validators.Validator):

    main_func: Callable[[pd.DataFrame], bool]

    def __init__(
        self,
        main_func: Callable[[pd.DataFrame], bool],
        mandatory: bool = True,
        description: str = None,
    ) -> None:
        self.main_func = main_func
        super().__init__(mandatory, description)

    # pylint: disable=arguments-renamed
    def evaluate(self, dataframe) -> Tuple[pd.DataFrame, validations.RootValidation]:

        self.validate_pandas_dataframe(dataframe)

        dataframe = dataframe.copy()

        try:

            validation = validations.RootValidation(self.description, self.mandatory)

            valid = self.main_func(self, dataframe)

            if not valid and self.amendment is not None:
                dataframe = self.amendment(dataframe)
                valid = self.main_func(self, dataframe)
                validation.amended = True

            validation.valid = True

            return dataframe, validation

        except Exception as error:
            raise validations.RootValidationError(
                self.description, self.mandatory, error
            ).with_traceback(error.__traceback__)

    def validate_pandas_dataframe(self, dataframe) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("A pandas.DataFrame object must be provided")

    def set_amendment(
        self, amendment: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> Type["RootValidator"]:
        self.amendment = amendment
        return self


class RootValidatorSet(validators.ValidatorSet):
    def validate(  # pylint: disable=arguments-renamed
        self, dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, validations.RootValidationSet]:

        dataframe = dataframe.copy()
        validation_set = validations.RootValidationSet()
        keep_validating = True

        for validator in self.validators:

            if keep_validating:
                try:
                    dataframe, validation = validator.evaluate(dataframe)
                except validations.RootValidationError as error:
                    validation = error
            else:
                validation = validations.SuspendedValidation(
                    validator.description, validator.mandatory
                )

            validation_set.add_validation(validation)
            if (
                (validation.valid is False and validator.mandatory)
                or isinstance(validation, validations.RootValidationError)
                or (not keep_validating)
            ):
                keep_validating = False

        return dataframe, validation_set
