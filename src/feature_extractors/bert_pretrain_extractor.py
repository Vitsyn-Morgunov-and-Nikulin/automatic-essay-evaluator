from typing import List

import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

from src.feature_extractors.base_extractor import BaseExtractor


class BertPretrainFeatureExtractor(BaseExtractor):
    """Extract [CLS] embedding feature from any untrained bert-like models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, model_name: str, max_length: int = 512, batch_size=64):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @torch.no_grad()
    def generate_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Generates features in batch-mode, obtained from untrained bert model.

        :param data: Series with full_text column
        :return: Dataframe, that have index - id's from data, and columns - bert features
        """
        torch.cuda.empty_cache()

        texts = data.tolist()
        self.model = self.model.to(self.device)

        classification_outputs = []
        for ii in trange(
                0, len(data), self.batch_size,
                total=len(data) // self.batch_size + 1,
                desc="Generating bert features..."
        ):
            text_batch = texts[ii: ii + self.batch_size]
            batch_encoded = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            output = self.model(**batch_encoded)
            cls_output = output['last_hidden_state'][:, 0].cpu()
            classification_outputs.append(cls_output)

        self.model = self.model.to("cpu")
        classification_outputs = torch.cat(classification_outputs, dim=0)
        torch.cuda.empty_cache()
        column_names = [f"{self.model_name}_feat_{ii}" for ii in range(len(classification_outputs[0]))]
        return pd.DataFrame(
            data=classification_outputs.tolist(),
            index=data.index,
            columns=column_names
        )


class ManyBertPretrainFeatureExtractor(BaseExtractor):
    def __init__(self, model_names: List[str], max_length: int = 512, batch_size=64):
        super(ManyBertPretrainFeatureExtractor, self).__init__()
        self.model_names = model_names
        self.max_length = max_length
        self.batch_size = batch_size

    def generate_features(self, X: pd.Series) -> pd.DataFrame:
        extractors = [
            BertPretrainFeatureExtractor(model_name, self.max_length, self.batch_size)
            for model_name in self.model_names
        ]
        dataframes = [
            extractor.generate_features(X) for extractor in extractors
        ]

        return pd.concat(dataframes, axis='columns')
