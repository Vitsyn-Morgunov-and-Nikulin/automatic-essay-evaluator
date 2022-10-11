from pathlib import Path
from typing import Optional, Union

import pandas as pd


class BaseSolution:
    """Base class for any competition solution."""

    def __init__(self, config: Optional[dict] = None):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def save(self, directory: Union[str, Path]) -> None:
        """Stores model to the directory.

        The directory must be empty.
        """
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def load(self, directory: Union[str, Path]) -> None:
        """Loads model from the directory.

        Initializes the solution correctly even if the config dict
        wasn't specified in the constructor.

        The directory should not contain files other than those produced by
        the `save` method of the same class.
        """
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")
