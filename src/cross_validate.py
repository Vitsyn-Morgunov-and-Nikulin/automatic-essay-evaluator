from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import KFold

from src.metrics import MSEMetric
from src.solutions.base_solution import BaseSolution
from src.utils import validate_x, validate_y


class CrossValidation:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, saving_dir: Path, n_splits: int = 5):
        self.k_fold = KFold(n_splits=n_splits)
        self.metric = MSEMetric()

        if not saving_dir.is_dir():
            saving_dir.mkdir(exist_ok=True)
        self.saving_dir = saving_dir
        self.base_solution: Optional[BaseSolution] = None

    def fit(self, model: BaseSolution, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Makes average fold prediction

        :param model: predictor from BaseSolution class
        :param X: Dataframe that has text_id and full_text columns
        :param y: Dataframe that has text_id, cohesion, ... columns
        :return: Dataframe with class scores for each split and overall CV score
        """

        validate_x(X)
        validate_y(y)

        scores = []
        self.base_solution = model
        for ii, (train_ind, test_ind) in enumerate(self.k_fold.split(X)):
            print(f"Training fold={ii}...")
            X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
            y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

            training_model = deepcopy(model)
            training_model.fit(X_train, y_train, val_X=X_test, val_y=y_test, fold=ii)

            y_pred = training_model.predict(X_test)
            class_rmse = self.metric.evaluate_class_rmse(y_pred, y_test)
            scores.append(class_rmse)

            training_model.save(self.saving_dir / f"cv_fold_{ii}")

            del training_model

        scores = pd.DataFrame(scores)
        mean_values = [scores.mean(axis='rows').values.tolist()]
        overall = pd.DataFrame(mean_values, columns=scores.columns, index=['overall'])

        scores = pd.concat([scores, overall], axis='rows')
        return scores

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Makes average fold prediction

        :param X: Dataframe that have text_id and full_text columns
        :return: prediction Dataframe that have text_id, cohesion, ... columns
        """
        assert list(self.saving_dir.iterdir()) is not [], "Cross validation is not trained yet"

        validate_x(X)

        predictions = []
        for ii in range(self.k_fold.n_splits):
            model_path = self.saving_dir / f"cv_fold_{ii}"

            model = deepcopy(self.base_solution)
            model.load(model_path)
            pred = model.predict(X)
            predictions.append(pred)

        mean_class_predictions = {}
        for column in ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']:
            values = [item[column].values for item in predictions]
            mean_pred = np.mean(values, axis=0)
            mean_class_predictions[column] = mean_pred

        mean_class_predictions = pd.DataFrame(mean_class_predictions)

        X = X.copy().drop(columns=['full_text'])
        X = pd.concat([X, mean_class_predictions], axis='columns')

        return X
