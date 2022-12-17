from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_reader import load_train_test_df
from src.metrics import MSEMetric
from src.solutions.base_solution import BaseSolution


class ConstantPredictorSolution(BaseSolution):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        submission_df = []

        for _, row in X.iterrows():
            submission_df.append({
                'text_id': row.text_id,
                'cohesion': 3.0,
                'syntax': 3.0,
                'vocabulary': 3.0,
                'phraseology': 3.0,
                'grammar': 3.0,
                'conventions': 3.0
            })

        return pd.DataFrame(submission_df)

    def save(self, directory: Union[str, Path]) -> None:
        pass

    def load(self, directory: Union[str, Path]) -> None:
        pass

    def to(self, device: str) -> 'BaseSolution':
        return self


def main():
    train_df, test_df = load_train_test_df()

    predictor = ConstantPredictorSolution()

    _, test_data = train_test_split(train_df, test_size=0.2)
    y_pred = predictor.predict(test_data)

    y_true = test_data[['text_id', 'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']]
    metric = MSEMetric()

    print(f"Calculation class metric: {metric.evaluate_class_rmse(y_pred, y_true)}")
    print(f"Calculation class metric: {metric.evaluate_class_rmse(y_pred, y_true)}")

    submission_df = predictor.predict(test_df)

    submission_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()
