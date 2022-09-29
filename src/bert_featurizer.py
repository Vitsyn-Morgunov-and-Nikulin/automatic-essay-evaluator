import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


class BertFeatureExtractor:
    """Extract [CLS] embedding feature from any untrained bert-like models"""

    def __init__(self, model_name: str):
        """
        :param model_name: name of model from hugging face
        """
        self.model_name = model_name

        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @torch.no_grad()
    def extract_features(self, row: pd.Series):
        text = row.full_text

        encoded = self.tokenizer(text=text, return_tensors='pt')
        output = self.model(**encoded)
        cls_output = output['last_hidden_state'][0][0]

        return pd.Series({
            f'{self.model_name}_feature_{ii}': cls_output[ii].item()
            for ii in range(len(cls_output))
        })
