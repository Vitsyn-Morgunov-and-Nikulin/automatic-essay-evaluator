import pandas as pd
from tqdm import tqdm

from src.text_cleaning.spell_checker import SmartSpellChecker
from src.text_cleaning.utils import preprocess_test


class TextPreprocessor:
    def __init__(self, spellcheck: SmartSpellChecker):
        self._spellcheck = spellcheck

    def preprocess_texts(self, data: pd.Series) -> pd.Series:
        out_texts = []

        for text in tqdm(data, desc="Preprocessing texts (correcting mistakes, removing tokens, etc.)..."):
            text = preprocess_test(text)
            text = self._spellcheck.correct_text(text)
            out_texts.append(text)

        return pd.Series(out_texts, index=data.index)
