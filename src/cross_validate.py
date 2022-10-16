from copy import deepcopy
from typing import List

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
