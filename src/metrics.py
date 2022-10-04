import pandas as pd
from sklearn.metrics import mean_squared_error


class MSEMetric:
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate_mse_class(y_pred: pd.DataFrame, y_true: pd.DataFrame) -> dict:
        result = {}

        for column in y_pred.columns:
            result[column] = mean_squared_error(y_pred[column], y_true[column], squared=False)

        return result

    def evaluate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame):
        result = self.evaluate_mse_class(y_pred, y_true)

        return sum(result.values()) / 6
