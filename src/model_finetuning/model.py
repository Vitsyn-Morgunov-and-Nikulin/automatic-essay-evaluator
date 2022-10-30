"""
    Copy pasted model from https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train/notebook
"""
import pytorch_lightning as pl
import torch

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup
import torch.nn as nn
from src.model_finetuning.metric import MCRMSELoss

from src.data_reader import load_train_test_df


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

        # freezing first 12 layers of DeBERTa from 24
        modules = [self.model.embeddings, self.model.encoder.layer[:20]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

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

        self.log('val/loss', loss)

        return {
            'loss': loss,
            'mc_rmse': loss
        }

    def validation_epoch_end(self, outputs):
        mean_mc_rmse = sum(output['mc_rmse'].item() for output in outputs) / len(outputs)
        self.log('val/epoch_loss', mean_mc_rmse)

    def configure_optimizers(self):
        weight_decay = self.config['weight_decay']
        lr = self.config['lr']

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': weight_decay,
                'name': 'group1'
            }, {
                'params':  [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'name': 'group2'
            }
        ]

        optimizer = AdamW(optimizer_parameters, lr=lr, weight_decay=0.0, eps=1e-6, betas=(0.9, 0.999))

        # epochs * int(len(train_data) * train_size / batch_size)
        train_steps_in_epoch = int(num_train_samples() * self.config['train_size'] / self.config['batch_size'])

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps_in_epoch * self.config['max_epochs'],
            num_cycles=0.5
        )
        return [optimizer], [scheduler]
