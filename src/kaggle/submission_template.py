from string import Template

source_code_template = Template("""
import os
from pathlib import Path
from typing import Any, List, Optional, Union

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


$implementation

def main():
    set_env_if_kaggle_environ()

    test_df_path = Path(os.environ['DATA_PATH']) / 'test.csv'
    if not test_df_path.is_file():
        raise OSError(f"File not found: {test_df_path.absolute()}")

    predictor = $predictor()

    test_df = pd.read_csv(test_df_path)
    submission_df = predictor.predict(test_df)

    submission_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()
""")


def get_source_code(predictor, implementation):
    source_code = source_code_template.substitute(predictor=predictor, implementation=implementation)
    return source_code
