from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from src.feature_extractors.text_statistics_utils import preprocess_test
from src.spell_checker import SmartSpellChecker
from src.text_preprocessings.base_preprocessing import BasePreprocessor


class SpellcheckTextPreprocessor(BasePreprocessor):
    def __init__(self, spellcheck: SmartSpellChecker):
        super(SpellcheckTextPreprocessor, self).__init__()
        self._spellcheck = spellcheck

    def preprocess_data(self, data: pd.Series) -> pd.Series:
        out_texts = []

        for text in tqdm(data, desc="Preprocessing texts (correcting mistakes, removing tokens, etc.)..."):
            text = self._preprocess_text(text)
            out_texts.append(text)

        return pd.Series(out_texts, index=data.index)

    @lru_cache
    def _preprocess_text(self, text: str) -> str:
        text = preprocess_test(text)
        text = self._spellcheck.correct_text(text)
        return text
