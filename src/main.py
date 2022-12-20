import os
import traceback
from distutils.dir_util import copy_tree

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import logging as transformer_log

from src.data_reader import load_train_test_df
from src.utils import (get_x_columns, pretty_cfg, report_to_telegram,
                       seed_everything)

seed_everything()
transformer_log.set_verbosity_error()


def run(cfg):
    predictor = instantiate(cfg.predictor)
    validator = instantiate(cfg.validator)

    train_df, test_df = load_train_test_df()
    x_columns = get_x_columns()
    train_x, train_y = train_df[x_columns], train_df.drop(columns=['full_text'])

    results = validator.fit(predictor, train_x, train_y)

    cv_mean = results.iloc[len(results) - 1].mean()
    print("CV results")
    print(results)
    print(f"CV mean: {cv_mean}")

    print(cfg.cwd)

    submission_df = validator.predict(test_df)
    submission_path = os.path.join(validator.saving_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    cv_results_path = os.path.join(validator.saving_dir, "cv_results.csv")
    results.to_csv(cv_results_path)

    weight_path = os.path.join(cfg.cwd, "data/weights")
    copy_tree(validator.saving_dir, weight_path)

    return cv_mean


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    try:
        metric = run(cfg)
        message = f"✅ Successful run from {cfg.timestamp}!\n\n"
        message += f"Metric: {metric:.5f} MCRMSE\n\n"
        message += f"Configuration:\n{pretty_cfg(cfg)}"
    except Exception:
        message = f"🚫 Run from {cfg.timestamp} failed!\n\n"
        message += traceback.format_exc()
    report_to_telegram(message)


if __name__ == "__main__":
    main()
