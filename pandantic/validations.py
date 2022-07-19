from typing import List, Optional


class Validation:

    description: str
    original_issues: Optional[int]
    pending_issues: Optional[int]
    valid: bool
    amended: bool
    additional_info: Optional[str]

    def __init__(self, description: str) -> None:
        self.description = description
        self.amended = False


class SuspendedValidation(Validation):
    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.additional_info = "Validation was suspended"


class ValidationError(Validation, Exception):
    def __init__(self, description: str, original_error: Exception) -> None:
        super().__init__(description)
        self.original_error = original_error


class ValidationSet:

    validations: List

    def __init__(self) -> None:
        self.validations = []

    def add_validation(self, validation: Validation):
        if not isinstance(validation, Validation):
            raise ValueError(f"Validation expected, got {type(validation)} instead.")

        self.validations.append(validation)
