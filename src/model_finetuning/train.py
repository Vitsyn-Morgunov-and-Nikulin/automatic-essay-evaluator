import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # noqa
from sklearn.model_selection import train_test_split

from src.data_reader import load_train_test_df
from src.model_finetuning.config import CONFIG
from src.model_finetuning.dataloader import ClassificationDataloader
from src.model_finetuning.losses import MCRMSELoss
from src.model_finetuning.model import BertLightningModel
from src.utils import get_target_columns, seed_everything

seed_everything()


def train(
        config: dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        verbose: bool = False
) -> BertLightningModel:
    log_dir = Path("logs/")
    log_dir.mkdir(exist_ok=True)

    model = BertLightningModel(config)

    dataloader = ClassificationDataloader(
        tokenizer=model.tokenizer,
        train_df=train_df,
        val_df=val_df,
        config=config
    )

    logger = WandbLogger(
        project="automated_essay_evaluator",
        entity="parmezano",
        config=config,
        log_model='all',
        group=os.environ['GROUP_NAME'],
    )
    wandb.run.log_code(".")
    wandb.watch(model, criterion=MCRMSELoss())

    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(
        dirpath=str(log_dir.resolve()),
        monitor='val/epoch_loss',
        verbose=True,
        mode='min',
        auto_insert_metric_name=True,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, model_checkpoint],
        accelerator=config['accelerator'],
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        precision=config['precision'],

        # this is for debug
        # max_epochs=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_predict_batches=1,
    )
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

    if verbose:
        print(f"Best class metric: {model.best_metric}, class scores: {model.class_metric}")

    model = BertLightningModel.load_from_checkpoint(model_checkpoint.best_model_path, config=config)
    return model


def predict(config: dict, model: BertLightningModel, df: pd.DataFrame) -> pd.DataFrame:
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        gradient_clip_val=config['gradient_clip_val'],
        precision=config['precision'],
    )

    predict_dataloader = ClassificationDataloader(
        tokenizer=model.tokenizer,
        train_df=df,
        val_df=df,
        config=config
    ).val_dataloader()

    validation_predictions = trainer.predict(model, predict_dataloader, return_predictions=True)
    validation_predictions = torch.vstack(validation_predictions)
    validation_predictions = pd.DataFrame({
        column: validation_predictions[:, ii] for ii, column in enumerate(get_target_columns())
    }, index=df.index)
    validation_predictions['text_id'] = df['text_id']

    return validation_predictions


def main():
    train_data, _ = load_train_test_df()
    train_df, val_df = train_test_split(train_data, train_size=CONFIG['train_size'])
    train(config=CONFIG, train_df=train_df, val_df=val_df, verbose=True)


if __name__ == '__main__':
    main()
