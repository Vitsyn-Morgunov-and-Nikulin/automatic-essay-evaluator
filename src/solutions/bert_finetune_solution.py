from pathlib import Path
from typing import Union

import pandas as pd
import pytorch_lightning as pl
import wandb
from transformers import logging

from src.cross_validate import CrossValidation
from src.data_reader import load_train_test_df
from src.model_finetuning.config import CONFIG
from src.model_finetuning.model import BertLightningModel
from src.model_finetuning.train import predict, train
from src.solutions.base_solution import BaseSolution
from src.utils import get_x_columns

logging.set_verbosity_error()


class BertFinetuningSolution(BaseSolution):
    model = None

    def __init__(self, config: dict):
        super(BertFinetuningSolution, self).__init__()
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        train_df = pd.concat([X, y], axis='columns')

        val_X, val_y, fold = kwargs['val_X'], kwargs['val_y'], kwargs['fold']
        val_df = pd.concat([val_X, val_y], axis='columns')

        self.model: BertLightningModel = train(self.config, train_df, val_df, fold=fold, verbose=False)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None, "Model is not trained yet"

        predictions = predict(self.config, self.model, X)
        return predictions

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True)

        trainer = pl.Trainer(accelerator=self.config['accelerator'])
        trainer.model = self.model
        trainer.save_checkpoint(directory / "lightning_model.ckpt", weights_only=True)

    def load(self, directory: Union[str, Path]) -> None:
        filepath = Path(directory) / "lightning_model.ckpt"

        if not filepath.is_file():
            raise OSError(f"File not found: {filepath.resolve()}")

        self.model = BertLightningModel.load_from_checkpoint(str(filepath), config=self.config)


def main():
    config = CONFIG
    saving_dir = Path("checkpoints/finetune_bert")
    train_df, test_df = load_train_test_df()

    x_columns = get_x_columns()
    train_x, train_y = train_df[x_columns], train_df.drop(columns=['full_text'])

    predictor = BertFinetuningSolution(config)
    cv = CrossValidation(saving_dir=saving_dir, n_splits=config['num_cross_val_splits'])

    results = cv.fit(predictor, train_x, train_y)
    print(f"CV metric: {results.iloc[len(results) - 1].mean()}")
    print("CV results")
    print(results)
    results.to_csv(saving_dir / "cv_results.csv")

    submission_df = cv.predict(test_df)
    submission_df.to_csv(saving_dir / "submission.csv", index=False)

    wandb.init(
        project="automated_essay_evaluator",
        entity="parmezano",
        group="train_deberta_model",
        name='weights_cv'
    )
    art = wandb.Artifact("bert-finetune-solution", type="model")
    art.add_dir(str(saving_dir.absolute()), name='data/')
    wandb.log_artifact(art)

    print("Finished training!")


if __name__ == '__main__':
    main()
