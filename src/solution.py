from pathlib import Path
from typing import Optional

import pandas as pd


class Solution:
    def __init__(self, config: Optional[dict] = None):
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def save(self, directory: str | Path):
        """Stores model to the directory.

        The directory must be empty
        """
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def load(self, directory: str | Path):
        """Loades model from the directory.

        Initializes the solution correctly even if the config dict
        wasn't specified in the constructor.

        The directory should not contain files other than those produced by
        the `save` method of the same class.
        """
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")
