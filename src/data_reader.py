import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def set_env_if_kaggle_environ() -> None:
    if 'KAGGLE_DATA_PROXY_TOKEN' in os.environ:
        os.environ['DATA_PATH'] = '/kaggle/input/feedback-prize-english-language-learning/'


def load_train_test_df(is_testing: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train/test dataframes

    :param is_testing: If set to true, load subsample of train/test dataframes
    :return Train and test dataframes

    """
    set_env_if_kaggle_environ()

    if is_testing:
        train_df_path = Path("src/tests/data/train_sample.csv")
        test_df_path = Path("src/tests/data/test_sample.csv")

    else:
        train_df_path = Path(os.environ['DATA_PATH']) / 'train.csv'
        test_df_path = Path(os.environ['DATA_PATH']) / 'test.csv'

    if not test_df_path.is_file():
        raise OSError(f"File not found: {test_df_path.absolute()}")

    if not train_df_path.is_file():
        raise OSError(f"File not found: {train_df_path.absolute()}")

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    return train_df, test_df
