from typing import Callable, T, Optional

import pandas as pd


def root_validator(
    pre=True, amendment: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
):
    def _decorator(func: Callable[[T, pd.DataFrame], bool]):
        func.root_validation = True
        func.pre = pre
        func.amendment = amendment
        func = classmethod(func)
        return func

    return _decorator
