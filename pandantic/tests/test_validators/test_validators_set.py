# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import validators, datatype_validators, shortcuts, validations


def test_validator_set_success():

    col = pd.Series(["1", "2", "3", "4", np.nan])

    validator_set = validators.ValidatorSet()
    non_null_validator = shortcuts.non_null().add_amendment(
        lambda column: column.fillna("5")
    )
    int_validator = datatype_validators.IntegerColumnValidator()
    below_three_validator = shortcuts.lower_or_equal_than(3)
    validator_set.add_validator(non_null_validator)
    validator_set.add_validator(int_validator)
    validator_set.add_validator(below_three_validator)

    column, validation_set = validator_set.validate(col)

    assert validation_set.validations[0].amended
    assert validation_set.validations[0].valid
    assert validation_set.validations[1].amended
    assert validation_set.validations[1].valid
    assert validation_set.validations[2].valid is False
    assert column.sum() == 15


def test_validator_set_suspended():

    col = pd.Series(["1", "2", "3", "4", np.nan])

    validator_set = validators.ValidatorSet()
    non_null_validator = shortcuts.non_null().add_amendment(
        lambda column: column.fillna("5")
    )
    int_validator = datatype_validators.IntegerColumnValidator()
    below_three_validator = shortcuts.lower_or_equal_than(3, mandatory=True)
    validator_set.add_validator(non_null_validator)
    validator_set.add_validator(int_validator)
    validator_set.add_validator(below_three_validator)
    validator_set.add_validator(below_three_validator)

    column, validation_set = validator_set.validate(col)

    assert validation_set.validations[0].amended
    assert validation_set.validations[0].valid
    assert validation_set.validations[1].amended
    assert validation_set.validations[1].valid
    assert validation_set.validations[2].valid is False
    assert isinstance(validation_set.validations[3], validations.SuspendedValidation)
    assert column.sum() == 15


def test_validator_set_error():

    col = pd.Series(["1", "2", "3", "4", np.nan])

    validator_set = validators.ValidatorSet()
    non_null_validator = shortcuts.non_null().add_amendment(
        lambda column: column.fillna("5")
    )
    below_three_validator = shortcuts.lower_or_equal_than(3, mandatory=True)
    validator_set.add_validator(non_null_validator)
    validator_set.add_validator(below_three_validator)

    _, validation_set = validator_set.validate(col)

    assert validation_set.validations[0].amended
    assert validation_set.validations[0].valid
    assert isinstance(validation_set.validations[1], validations.ValidationError)
