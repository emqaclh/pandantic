from typing import Optional

from pandantic import validations


class ColumnEvaluation:

    validation_set: Optional[validations.ValidationSet]
    valid: Optional[bool]
    amended: Optional[bool]
    warnings: Optional[bool]

    def __init__(
        self, validation_set: Optional[validations.ValidationSet] = None
    ) -> None:
        self.validation_set = validation_set
        if self.validation_set is not None:
            self.check_validations()
        else:
            self.valid = self.amended = self.warnings = None

    def check_validations(self) -> None:
        self.valid = all(
            [
                validation.valid
                for validation in self.validation_set
                if validation.mandatory
            ]
        )
        self.amended = any([validation.amended for validation in self.validation_set])
        self.warnings = any(
            [
                not validation.valid
                for validation in self.validation_set
                if not validation.mandatory
            ]
        )


class SuspendedColumnEvaluation(ColumnEvaluation):
    pass


class MissingColumn(ColumnEvaluation):
    pass


class UnhandledColumn(ColumnEvaluation):
    pass
