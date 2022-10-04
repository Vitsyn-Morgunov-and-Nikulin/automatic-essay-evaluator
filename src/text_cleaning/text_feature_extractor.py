from functools import reduce
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.text_cleaning.spell_checker import SmartSpellChecker
from src.text_cleaning.utils import (count_how_many_words_are_repeating,
                                     count_misspelled_words, count_punctuation,
                                     count_words, preprocess_test)


class TextFeatureExtractor:
    def __init__(self):
        self._spellcheck = SmartSpellChecker()

    def _generate_features(self, raw_text: str):
        preprocessed_text = preprocess_test(raw_text)
        cleaned_text = self._spellcheck.correct_text(preprocessed_text)

        features: List[Dict] = [
            count_punctuation(raw_text),
            count_misspelled_words(preprocessed_text, self._spellcheck),
            count_words(cleaned_text),
            count_how_many_words_are_repeating(cleaned_text)
        ]

        return reduce(lambda x, y: {**x, **y}, features)

    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        features = [self._generate_features(text) for text in tqdm(data, desc="Gen. text features...")]
        return pd.DataFrame(features, index=data.index)

    def preprocess_texts(self, data: pd.Series):
        out_texts = []

        for text in tqdm(data, desc="Preprocessing texts..."):
            text = preprocess_test(text)
            text = self._spellcheck.correct_text(text)
            out_texts.append(text)

        return pd.Series(out_texts, index=data.index)
