import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer


class BertFeatureExtractor:
    """Extract [CLS] embedding feature from any untrained bert-like models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, model_name: str, max_length: int = 512, batch_size=64):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @torch.no_grad()
    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Generates features in batch-mode, obtained from untrained bert model.

        :param data: Series, that have index - id's from train/test dataframe and contains raw texts
        :return: Dataframe, that have index - id's from data, and columns - bert features, and
        """

        classification_outputs = []
        texts = data.tolist()
        for ii in trange(
                0, len(texts), self.batch_size,
                total=len(texts) // self.batch_size + 1,
                desc="Generating bert features..."
        ):
            text = texts[ii: ii + self.batch_size]
            batch_encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            output = self.model(**batch_encoded)
            cls_output = output['last_hidden_state'][:, 0].cpu()
            classification_outputs.append(cls_output)

        classification_outputs = torch.cat(classification_outputs, dim=0)

        column_names = [f"{self.model_name}_feat_{ii}" for ii in range(len(classification_outputs[0]))]

        return pd.DataFrame(
            data=classification_outputs.tolist(),
            index=data.index,
            columns=column_names
        )
