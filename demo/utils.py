import pandas as pd
import streamlit as st
import torch

from src.model_finetuning.config import CONFIG
from src.model_finetuning.model import BertLightningModel
from src.utils import get_target_columns


@st.cache(allow_output_mutation=True)
def load_model() -> BertLightningModel:

    ckpt_path = "demo/model.ckpt"
    model = BertLightningModel.load_from_checkpoint(ckpt_path, config=CONFIG, map_location='cpu')

    return model


@torch.no_grad()
def process_text(_text: str, _model: BertLightningModel) -> pd.DataFrame:
    tokens = _model.tokenizer([_text], return_tensors='pt')
    outputs = _model(tokens)[0].tolist()

    df = pd.DataFrame({
        'Criterion': get_target_columns(),
        'Grade': outputs
    })

    return df
