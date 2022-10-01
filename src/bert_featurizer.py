import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


class BertFeatureExtractor:
    """Extract [CLS] embedding feature from any untrained bert-like models"""

    def __init__(self, model_name: str, padding: str = 'max_length', max_length: int = 512):
        self.model_name = model_name
        self.padding = padding
        self.max_length = max_length

        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @torch.no_grad()
    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Generates features, obtained from untrained bert model.

        :param data: Series, that have index - id's from train/test dataframe and contains raw texts
        :return: Dataframe, that have index - id's from data, and columns - bert features, and
        """
        texts = data.tolist()

        encoded = self.tokenizer(text=texts, padding=self.padding, max_length=self.max_length, return_tensors='pt')
        output = self.model(**encoded)
        cls_output = output['last_hidden_state'][:, 0]

        column_names = [f"{self.model_name}_feat_{ii}" for ii in range(len(cls_output[0]))]

        return pd.DataFrame(
            data=cls_output.tolist(),
            index=data.index,
            columns=column_names
        )
