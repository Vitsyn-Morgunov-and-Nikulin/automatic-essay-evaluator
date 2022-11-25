import os
import random
import string
from typing import List
import requests
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv

load_dotenv()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pandas_set_print_options():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def get_target_columns() -> List[str]:
    return ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


def get_x_columns() -> List[str]:
    return ['text_id', 'full_text']


def validate_x(X: pd.DataFrame) -> None:
    columns = set(X.columns)

    if len(columns) != 2 or any(col not in get_x_columns() for col in columns):
        print(X)
        raise RuntimeError(f"X has incorrect columns: it should contain only {get_x_columns()}")


def validate_y(y: pd.DataFrame) -> None:
    columns = set(y.columns)

    y_needed_columns = get_target_columns() + ['text_id']

    if len(columns) != 7 or any(col not in y_needed_columns for col in columns):
        print(y)
        raise RuntimeError(f"y has incorrect columns: it should contain only {y_needed_columns}")


def get_random_string(length) -> str:
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def report_to_telegram(cfg, metric=None, chat_ids=(481338688, )):
    for chat_id in chat_ids:
        requests.get(
            'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text}'.format(
                bot_token=os.environ['BOT_TOKEN'],
                chat_id=chat_id,
                text=f'Finished training \n{HydraConfig.get().overrides.task}\n{OmegaConf.to_yaml(cfg, resolve=True)}\nCV mean: {metric}')
        )
