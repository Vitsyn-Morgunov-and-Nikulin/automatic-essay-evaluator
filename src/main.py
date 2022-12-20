import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import logging as transformer_log

from src.data_reader import load_train_test_df
from src.utils import (get_x_columns, pretty_cfg, report_to_telegram,
                       save_experiment, seed_everything)

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

    submission_df = validator.predict(test_df)

    save_experiment(cfg, submission_df, results, validator.saving_dir)

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
