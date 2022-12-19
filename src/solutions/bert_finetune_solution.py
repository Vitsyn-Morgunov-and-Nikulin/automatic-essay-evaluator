import os
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
from src.utils import get_random_string, get_x_columns

logging.set_verbosity_error()


os.environ['GROUP_NAME'] = 'train_deberta_model-' + get_random_string(6)


class BertFinetuningPredictor(BaseSolution):

    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_classes=6,
        lr=2e-5,
        batch_size=8,
        num_workers=8,
        max_length=512,
        weight_decay=0.01,
        accelerator='gpu',
        max_epochs=5,
        accumulate_grad_batches=4,
        precision=16,
        gradient_clip_val=1000,
        train_size=0.8,
        num_cross_val_splits=5,
        num_frozen_layers=20,
    ):
        super(BertFinetuningPredictor, self).__init__()
        self.config = dict(
            model_name=model_name,
            num_classes=num_classes,
            lr=lr,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            weight_decay=weight_decay,
            accelerator=accelerator,
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            train_size=train_size,
            num_cross_val_splits=num_cross_val_splits,
            num_frozen_layers=num_frozen_layers,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        train_df = pd.concat([X, y], axis='columns')

        val_X, val_y, fold = kwargs['val_X'], kwargs['val_y'], kwargs['fold']
        val_df = pd.concat([val_X, val_y], axis='columns')

        self.model: BertLightningModel = train(self.config, train_df, val_df, verbose=False)

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
        # trainer.save_checkpoint(directory / "lightning_model.ckpt", weights_only=True)

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

    predictor = BertFinetuningPredictor(config)
    cv = CrossValidation(saving_dir=str(saving_dir), n_splits=config['num_cross_val_splits'])

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
        group=os.environ['GROUP_NAME'],
        name='weights_cv'
    )
    art = wandb.Artifact("bert-finetune-solution", type="model")
    art.add_dir(str(saving_dir.absolute()), name='data/')
    wandb.log_artifact(art)

    print("Finished training!")


if __name__ == '__main__':
    main()
