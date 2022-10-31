"""
    Copy pasted model from https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train/notebook
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.data_reader import load_train_test_df
from src.model_finetuning.metric import MCRMSELoss


def num_train_samples():
    train_df, _ = load_train_test_df()
    return len(train_df)


class MeanPooling(nn.Module):
    # taking mean of last hidden state with mask

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class BertLightningModel(pl.LightningModule):

    def __init__(self, config: dict):
        super(BertLightningModel, self).__init__()

        self.config = config

        huggingface_config = AutoConfig.from_pretrained(self.config['model_name'], output_hidden_states=True)
        huggingface_config.hidden_dropout = 0.
        huggingface_config.hidden_dropout_prob = 0.
        huggingface_config.attention_dropout = 0.
        huggingface_config.attention_probs_dropout_prob = 0.

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = AutoModel.from_pretrained(self.config['model_name'], config=huggingface_config)

        self.pool = MeanPooling()

        self.fc = nn.Linear(in_features=1024, out_features=6)

        self.loss = MCRMSELoss()

        # freezing first 20 layers of DeBERTa from 24
        modules = [self.model.embeddings, self.model.encoder.layer[:self.config['num_frozen_layers']]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        self.class_metric = None
        self.best_metric = None

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        bert_features = self.pool(last_hidden_state, inputs['attention_mask'])

        logits = self.fc(bert_features)

        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch
        labels = inputs.pop("labels", None)
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.log('train/loss', loss)

        return {
            'loss': loss,
            'mc_rmse': loss
        }

    def training_epoch_end(self, outputs):
        mean_mc_rmse = sum(output['mc_rmse'].item() for output in outputs) / len(outputs)
        self.log("train/epoch_loss", mean_mc_rmse)

    def validation_step(self, batch, batch_idx):
        inputs = batch
        labels = inputs.pop("labels", None)
        logits = self(inputs)
        loss = self.loss(logits, labels)
        class_rmse = self.loss.class_mcrmse(logits, labels)

        self.log('val/loss', loss)

        return {
            'loss': loss,
            'mc_rmse': loss,
            'class_mc_rmse': class_rmse
        }

    def validation_epoch_end(self, outputs):
        mean_mc_rmse = sum(output['mc_rmse'].item() for output in outputs) / len(outputs)
        class_metrics = torch.stack([output['class_mc_rmse'] for output in outputs]).mean(0).tolist()
        class_metrics = [round(item, 4) for item in class_metrics]
        self.log('val/epoch_loss', mean_mc_rmse)

        if self.best_metric is None or mean_mc_rmse < self.best_metric:
            self.best_metric = mean_mc_rmse
            self.class_metric = class_metrics

    def configure_optimizers(self):
        # weight_decay = self.config['weight_decay']
        lr = self.config['lr']

        # In original solution authors add weight decaying to some parameters

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=0.0, eps=1e-6, betas=(0.9, 0.999))

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['max_epochs'],
        )
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs = batch
        inputs.pop("labels", None)
        logits = self(inputs)

        return logits
