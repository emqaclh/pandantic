# pylint: disable=unused-import
import pytest

import pandas as pd
import numpy as np

from pandantic import shortcuts


def test_range_shortcuts():

    col = pd.Series([1, 2, 3])

    _, original_issue_count, _, _, _ = shortcuts.between_range(0, 5).evaluate(col)
    assert not original_issue_count

    _, original_issue_count, _, _, _ = shortcuts.greater_than(1).evaluate(col)
    assert original_issue_count == 1

    _, original_issue_count, _, _, _ = shortcuts.greater_or_equal_than(1).evaluate(col)
    assert not original_issue_count

    _, original_issue_count, _, _, _ = shortcuts.lower_than(2).evaluate(col)
    assert original_issue_count == 2

    _, original_issue_count, _, _, _ = shortcuts.lower_or_equal_than(2).evaluate(col)
    assert original_issue_count == 1


def test_category_shortcut():

    col = pd.Series(["a", "b", "c"])

    _, original_issue_count, _, _, _ = shortcuts.in_categories(["a", "b"]).evaluate(col)
    assert original_issue_count == 1


def test_non_null_shortcut():

    col = pd.Series(["a", "b", np.nan, "c", np.nan])

    _, original_issue_count, _, _, _ = shortcuts.non_null().evaluate(col)
    assert original_issue_count == 2


def test_is_unique_shortcut():

    col = pd.Series(["a", "b", "c", "b", "a"])

    _, original_issue_count, _, _, _ = shortcuts.is_unique().evaluate(col)
    assert original_issue_count == 2


def test_match_pattern_shortcut():

    col = pd.Series(["a", "b", "c_", "b_", "a", np.nan])

    _, original_issue_count, _, _, _ = shortcuts.match_pattern(r".{1}").evaluate(col)
    assert original_issue_count == 2
