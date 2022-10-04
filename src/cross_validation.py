from typing import List

import pandas as pd
from sklearn.model_selection import KFold


class CrossValidationBase:
    def __init__(self, *args, **kwargs):
        pass

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> List:
        pass

    def predict(self, predictions: List[pd.DataFrame]) -> pd.DataFrame:
        pass


class CrossValidation(CrossValidationBase):
    def __init__(self, n_splits=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits)

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> List:
        splits = []

        for train_index, test_index in self.kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    def predict(self, predictions: List[pd.DataFrame]) -> pd.DataFrame:
        return sum(predictions) / len(predictions)
