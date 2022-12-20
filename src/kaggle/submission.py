
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def set_env_if_kaggle_environ():
    if 'KAGGLE_DATA_PROXY_TOKEN' in os.environ:
        os.environ['DATA_PATH'] = '/kaggle/input/feedback-prize-english-language-learning/'


class BaseSolution:

    def __init__(self, config: Optional[dict] = None):
        self.models: List[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def save(self, directory: Union[str, Path]) -> None:
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")

    def load(self, directory: Union[str, Path]) -> None:
        raise NotImplementedError(f"Abstract class {type(self).__name__} is used")


class ConstantPredictorSolution(BaseSolution):
    def __init__(self, const=3.0):
        super().__init__()
        self.const = const

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        submission_df = []

        for _, row in X.iterrows():
            submission_df.append({
                'text_id': row.text_id,
                'cohesion': self.const,
                'syntax': self.const,
                'vocabulary': self.const,
                'phraseology': self.const,
                'grammar': self.const,
                'conventions': self.const
            })

        return pd.DataFrame(submission_df)

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)

        path = directory / "weights.ckpt"
        with open(path, 'w') as file:
            file.write(str(self.const))

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)

        path = directory / "weights.ckpt"
        with open(path, 'r') as file:
            self.const = float(file.read())

    def to(self, device: str) -> 'BaseSolution':
        return self


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
