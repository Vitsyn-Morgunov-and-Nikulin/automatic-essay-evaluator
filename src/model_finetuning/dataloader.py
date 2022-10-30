import torch

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from src.utils import get_target_columns
import pytorch_lightning as pl


def collate_fn(data):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []

    for item in data:
        input_ids.append(item['input_ids'].squeeze())
        token_type_ids.append(item['token_type_ids'].squeeze())
        attention_mask.append(item['attention_mask'].squeeze())
        labels.append(item['labels'].squeeze())

    return {
        "input_ids": torch.stack(input_ids),
        'token_type_ids': torch.stack(token_type_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, df: pd.DataFrame, config: dict):
        self.config = config
        self.tokenizer = tokenizer

        self.df = df

        self.features = self.tokenizer(
            text=df.full_text.tolist(),
            max_length=self.config['max_length'],
            padding=True,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt',
        )

        self.features['labels'] = torch.as_tensor(df[get_target_columns()].values, dtype=torch.float32)

    def __getitem__(self, item):
        """Returns dict with input_ids, token_type_ids, attention_mask, labels
        """
        return {
            'input_ids': self.features['input_ids'][item],
            'token_type_ids': self.features['token_type_ids'][item],
            'attention_mask': self.features['attention_mask'][item],
            'labels': self.features['labels'][item]
        }

    def __len__(self):
        return len(self.df)


class ClassificationDataloader(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: BertTokenizer,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            config: dict
    ):
        super().__init__()
        self.config = config

        self.train_data = ClassificationDataset(tokenizer, train_df, config)
        self.val_data = ClassificationDataset(tokenizer, val_df, config)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn
        )
