from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import KFold

from src.metrics import MSEMetric
from src.solutions.base_solution import BaseSolution
from src.utils import validate_x, validate_y


class CrossValidation:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, saving_dir: str, n_splits: int = 5):
        _saving_dir = Path(saving_dir)

        self.k_fold = KFold(n_splits=n_splits)
        self.metric = MSEMetric()

        if not _saving_dir.is_dir():
            _saving_dir.mkdir(exist_ok=True, parents=True)
        self.saving_dir = _saving_dir
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

        _scores = pd.DataFrame(scores)
        mean_values = [_scores.mean(axis='rows').values.tolist()]
        overall = pd.DataFrame(mean_values, columns=_scores.columns, index=['overall'])

        _scores = pd.concat([_scores, overall], axis='rows')
        return _scores

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

            if not self.base_solution:
                raise TypeError
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

    def save(self, path: Union[str, Path]):
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True)

        if not self.base_solution or not self.base_solution.models:
            raise TypeError

        for ii, model in enumerate(self.base_solution.models):
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

            if not self.base_solution or not self.base_solution.models:
                raise TypeError

            self.base_solution.models.append(predictor_copy)

        print(f"Loaded model successfully from: {path.resolve()}.")
