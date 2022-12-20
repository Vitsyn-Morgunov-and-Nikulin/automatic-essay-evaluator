import json
import os
import random
import shutil
import string
from distutils.dir_util import copy_tree
from typing import List

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

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


def report_to_telegram(message):
    requests.get(
        'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text}'.format(
            bot_token=os.environ['BOT_TOKEN'],
            chat_id=os.environ['CHAT_ID'],
            text=message)
    )


def pretty_cfg(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_dict, indent=2)
    return cfg_json


def save_experiment(cfg, submission_df, results, saving_dir):
    submission_path = os.path.join(saving_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    cv_results_path = os.path.join(saving_dir, "cv_results.csv")
    results.to_csv(cv_results_path)

    weight_path = os.path.join(cfg.cwd, "data/weights")
    copy_tree(saving_dir, weight_path)

    src_config = os.path.join(".hydra", "config.yaml")
    dst_config = os.path.join(weight_path, "config.yaml")
    shutil.copy(src_config, dst_config)
