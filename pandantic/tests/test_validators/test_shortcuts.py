# pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest

from pandantic import shortcuts


def test_range_shortcuts():

    col = pd.Series([1, 2, 3])

    _, validation = shortcuts.between_range(0, 5).evaluate(col)
    assert not validation.original_issues

    _, validation = shortcuts.greater_than(1).evaluate(col)
    assert validation.original_issues == 1

    _, validation = shortcuts.greater_or_equal_than(1).evaluate(col)
    assert not validation.original_issues

    _, validation = shortcuts.lower_than(2).evaluate(col)
    assert validation.original_issues == 2

    _, validation = shortcuts.lower_or_equal_than(2).evaluate(col)
    assert validation.original_issues == 1


def test_category_shortcut():

    col = pd.Series(["a", "b", "c"])

    _, validation = shortcuts.in_categories(["a", "b"]).evaluate(col)
    assert validation.original_issues == 1


def test_non_null_shortcut():

    col = pd.Series(["a", "b", np.nan, "c", np.nan])

    _, validation = shortcuts.non_null().evaluate(col)
    assert validation.original_issues == 2


def test_is_unique_shortcut():

    col = pd.Series(["a", "b", "c", "b", "a"])

    _, validation = shortcuts.is_unique().evaluate(col)
    assert validation.original_issues == 2


def test_match_pattern_shortcut():

    col = pd.Series(["a", "b", "c_", "b_", "a", np.nan])

    _, validation = shortcuts.match_pattern(r".{1}").evaluate(col)
    assert validation.original_issues == 2
