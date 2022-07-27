from typing import List, Optional, Iterator


class Validation:

    description: str
    original_issues: Optional[int]
    pending_issues: Optional[int]
    valid: bool
    amended: bool
    mandatory: bool
    additional_info: Optional[str]

    def __init__(self, description: str, mandatory: bool) -> None:
        self.description = description
        self.mandatory = mandatory
        self.amended = False
        self.valid = False
        self.original_issues = None
        self.pending_issues = None
        self.additional_info = None


class SuspendedValidation(Validation):
    def __init__(self, description: str, mandatory: bool) -> None:
        super().__init__(description, mandatory)
        self.additional_info = "Validation was suspended"


class RootValidation(Validation):
    pass


class ValidationError(Validation, Exception):
    def __init__(
        self, description: str, mandatory: bool, original_error: Exception
    ) -> None:
        super().__init__(description, mandatory)
        self.original_error = original_error


class RootValidationError(ValidationError):
    pass


class ValidationSet:

    validations: List[Validation]

    def __init__(self) -> None:
        self.validations = []

    def __iter__(self) -> Iterator[Validation]:
        return iter(self.validations)

    def add_validation(self, validation: Validation):
        if not isinstance(validation, Validation):
            raise ValueError(f"Validation expected, got {type(validation)} instead.")

        self.validations.append(validation)


class RootValidationSet(ValidationSet):
    pass
