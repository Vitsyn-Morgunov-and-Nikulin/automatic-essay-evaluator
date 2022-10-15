from copy import deepcopy
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import KFold

from src.metrics import MSEMetric
from src.solutions.base_solution import BaseSolution
from src.utils import validate_x, validate_y


class CrossValidation:
    models: List[BaseSolution] = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_splits: int = 5):
        self.k_fold = KFold(n_splits=n_splits)
        self.metric = MSEMetric()

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

        for ii, (train_ind, test_ind) in enumerate(self.k_fold.split(X)):
            print(f"Training fold={ii}...")
            X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
            y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

            training_model = deepcopy(model)
            training_model.fit(X_train, y_train)

            y_pred = training_model.predict(X_test)
            class_rmse = self.metric.evaluate_class_rmse(y_pred, y_test)
            scores.append(class_rmse)

            self.models.append(training_model)

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
        assert self.models is not [], "Cross validation is not trained yet"

        validate_x(X)

        predictions = []
        for model in self.models:
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

    def save(self, path: Union[str, Path]):
        assert self.models is not [], "Models should be trained before saving them"

        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True)

        for ii, model in enumerate(self.models):
            cv_model_path = path / f"cv_fold_{ii}"
            model.save(cv_model_path)

        print(f"Saved weights successfully to: {path.resolve()}.")

    def load(self, path: Union[str, Path], predictor: BaseSolution):
        path = Path(path)

        assert path.is_dir(), f"Weights dir. not exists: {path.resolve()}"

        for ii in range(self.k_fold.n_splits):
            cv_model_path = path / f"cv_fold_{ii}"

            assert cv_model_path.is_dir(), f"Dir. with fold={ii} not exists: {cv_model_path.resolve()}"

            predictor_copy = deepcopy(predictor)
            predictor_copy.load(cv_model_path)
            self.models.append(predictor_copy)

        print(f"Loaded model successfully from: {path.resolve()}.")
