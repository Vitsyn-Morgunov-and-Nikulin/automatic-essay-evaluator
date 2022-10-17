from collections import defaultdict
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd

from src.feature_extractors.base_extractor import BaseExtractor


class TermFrequencyFeatureExtractor(BaseExtractor):
    """Build a dataframe with a distribution of term frequencies

    Usage:
        >>> data = pd.read_csv("data/raw/train.csv").set_index("text_id")
        >>> featurizer = TermFrequencyFeaturizer()
        >>> X = featurizer.featurize(data.full_text)
        >>> y = data["vocabulary"]
        >>> model = catboost.CatBoostRegressor()
        >>> model.fit(x_train, y_train)

    Possible improvements:
        - Add word corrections: triying -> trying
        - Count not only word frequencies, but number of unique words in each hist bin
    """

    MAX_TERM_FREQUENCY = 23135751162  # it's so in the term frequency dataset

    def __init__(self, n_bins: int = 40):
        self.term2freq: Dict[str, int] = self._load_term2freq_dict()
        self.bins = self._make_bins(n_bins)
        self.feature_names = [
            f"bin_{round(self.bins[i], 1)}_{round(self.bins[i+1], 1)}"
            for i in range(len(self.bins) - 1)
        ]
        nltk.download("punkt")

    def _make_bins(self, n_bins: int) -> np.ndarray:
        min_bin = 0
        max_bin = np.log1p(self.MAX_TERM_FREQUENCY)
        bins = np.linspace(min_bin, max_bin, n_bins)
        return bins

    def _load_term2freq_dict(self) -> Dict[str, int]:
        term_frequencies = pd.read_csv("data/word_frequencies/unigram_freq.csv")
        term2freq: Dict[str, int] = defaultdict(lambda: 0)
        term2freq.update(term_frequencies.set_index("word").to_dict()["count"])
        return term2freq

    def generate_features(self, texts: pd.Series) -> pd.DataFrame:
        """Extracts features from the text in the form of histogram of word frequencies

        Logarithm operation is applied to the frequencies for the sake of distribution
        normality.
        """
        feature_df = texts.apply(self._compute_word_frequency_histogram)
        feature_df.columns = self.feature_names
        return feature_df

    def _compute_word_frequency_histogram(self, text: str) -> pd.Series:
        term_frequencies: List[int] = self._compute_term_frequencies_from_text(text)
        histogram_values: np.ndarray = self._build_histogram(term_frequencies)
        return pd.Series(histogram_values)

    def _compute_term_frequencies_from_text(self, text: str) -> List[int]:
        tokens = nltk.tokenize.word_tokenize(text)
        words = [token.lower() for token in tokens if token.isalpha()]
        word_frequencies = [self.term2freq[word] for word in words]
        return word_frequencies

    def _build_histogram(self, values: List[int]) -> np.ndarray:
        values_log = np.log1p(values)
        histogram, __ = np.histogram(values_log, bins=self.bins)
        normalized_histogram = histogram / len(values)
        return normalized_histogram
