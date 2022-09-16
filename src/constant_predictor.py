import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv

from src.solution import Solution

load_dotenv()


def set_env_if_kaggle_environ():
    if 'KAGGLE_DATA_PROXY_TOKEN' in os.environ:
        os.environ['DATA_PATH'] = '/kaggle/input/feedback-prize-english-language-learning/'


class ConstantPredictorSolution(Solution):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
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

    def save(self, directory: Union[str, Path]):
        pass

    def load(self, directory: Union[str, Path]):
        pass


def main():
    set_env_if_kaggle_environ()

    test_df_path = Path(os.environ['DATA_PATH']) / 'test.csv'
    if not test_df_path.is_file():
        raise OSError(f"File not found: {test_df_path.absolute()}")

    predictor = ConstantPredictorSolution()

    test_df = pd.read_csv(test_df_path)
    submission_df = predictor.predict(test_df)

    submission_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()
