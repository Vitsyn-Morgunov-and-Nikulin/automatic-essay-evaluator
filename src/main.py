import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import logging as transformer_log

from src.data_reader import load_train_test_df
from src.utils import get_x_columns, report_to_telegram, seed_everything

seed_everything()
transformer_log.set_verbosity_error()


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    predictor = instantiate(cfg.predictor)
    validator = instantiate(cfg.validator)

    train_df, test_df = load_train_test_df()
    x_columns = get_x_columns()
    train_x, train_y = train_df[x_columns], train_df.drop(columns=['full_text'])

    results = validator.fit(predictor, train_x, train_y)
    print("CV results")
    print(results)

    cv_mean = results.iloc[len(results) - 1].mean()
    print(f"CV mean: {cv_mean}")

    submission_df = validator.predict(test_df)
    submission_path = os.path.join(validator.saving_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    cv_results_path = os.path.join(validator.saving_dir, "cv_results.csv")
    results.to_csv(cv_results_path)

    report_to_telegram(cfg=cfg, metric=cv_mean)


if __name__ == "__main__":
    main()
