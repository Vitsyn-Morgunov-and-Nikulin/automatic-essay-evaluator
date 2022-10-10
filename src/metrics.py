from typing import Dict

import pandas as pd
from sklearn.metrics import mean_squared_error


class MSEMetric:
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate_class_rmse(y_pred: pd.DataFrame, y_true: pd.DataFrame) -> Dict[str, float]:
        result = {}

        for column in y_pred.drop(columns=['text_id']).columns:
            result[column] = mean_squared_error(y_pred[column], y_true[column], squared=False)

        return result

    def evaluate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
        result = self.evaluate_class_rmse(y_pred, y_true)

        return sum(result.values()) / len(result)
