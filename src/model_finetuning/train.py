from pathlib import Path

from pytorch_lightning.loggers import WandbLogger  # noqa
from sklearn.model_selection import train_test_split

from src.data_reader import load_train_test_df
from src.model_finetuning.config import CONFIG
from src.model_finetuning.model import BertLightningModel
from src.model_finetuning.dataloader import ClassificationDataloader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.utils import seed_everything
from src.model_finetuning.metric import MCRMSELoss

seed_everything()


def main():
    log_dir = Path("logs/")
    log_dir.mkdir(exist_ok=True)

    train_data, test_data = load_train_test_df()
    train_df, val_df = train_test_split(train_data, train_size=CONFIG['train_size'])

    model = BertLightningModel(CONFIG)

    dataloader = ClassificationDataloader(
        tokenizer=model.tokenizer,
        train_df=train_df,
        val_df=val_df,
        config=CONFIG
    )

    logger = WandbLogger(
        project="automated_essay_evaluator",
        entity="parmezano",
        config=CONFIG,
        log_model='all',
        name="train_deberta_model"
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
        accelerator=CONFIG['accelerator'],
        max_epochs=CONFIG['max_epochs'],
        accumulate_grad_batches=CONFIG['accumulate_grad_batches'],
        gradient_clip_val=CONFIG['gradient_clip_val'],
        precision=CONFIG['precision']
    )

    trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())
    wandb.save(model_checkpoint.best_model_path)
    print(f"best val metric: {model_checkpoint.best_model_score}")
    print("Finished!")


if __name__ == '__main__':
    main()
