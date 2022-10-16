from typing import Dict, List
import pandas as pd
import numpy as np
import tqdm
import nltk
from collections import defaultdict
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# TODO: add word corrections: triying -> trying
class TermFrequencyFeaturizer:
    MAX_TERM_FREQ = 23135751162

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.term2freq: Dict[str, int] = self._load_term2freq_dict()
        nltk.download("punkt")

    def _load_term2freq_dict(self) -> Dict[str, int]:
        term_frequencies = pd.read_csv("data/word_frequencies/unigram_freq.csv")
        term2freq: Dict[str, int] = defaultdict(lambda: 0)
        term2freq.update(term_frequencies.set_index("word").to_dict()['count'])
        return term2freq

    def featurize(self, texts: pd.Series, n_bins: int=40) -> pd.DataFrame:
        """Extracts features from the text in the form of histogram of word frequencies

        Logarithm operation is applied to the frequencies for the sake of distribution
        normality.
        """
        bins = self._make_bins(n_bins)
        feature_df = texts.apply(self._compute_word_frequency_histogram, bins=bins)
        feature_df.columns = [
            f"bin_{round(bins[i], 1)}_{round(bins[i+1], 1)}"
            for i in range(len(bins) - 1)
        ]
        return feature_df

    def _make_bins(self, n_bins: int) -> np.ndarray:
        min_bin = 0
        max_bin = np.log1p(self.MAX_TERM_FREQ)
        bins = np.linspace(min_bin, max_bin, n_bins)
        return bins

    def _compute_word_frequency_histogram(self, text: str, bins: np.ndarray) -> pd.Series:
        term_frequencies: List[int] = self._compute_term_frequencies_from_text(text)
        histogram_values: np.ndarray = self._build_histogram(term_frequencies, bins)
        return pd.Series(histogram_values)

    def _compute_term_frequencies_from_text(self, text: str) -> List[int]:
        tokens = nltk.tokenize.word_tokenize(text)
        words = [token.lower() for token in tokens if token.isalpha()]
        word_frequencies = [self.term2freq[word] for word in words]
        return word_frequencies

    def _build_histogram(self, values: List[int], bins: np.ndarray) -> np.ndarray:
        values_log = np.log1p(values)
        histogram, __ = np.histogram(values_log, bins=bins)
        normalized_histogram = (histogram / len(values))
        return normalized_histogram


if __name__ == "__main__":
    data = pd.read_csv("data/raw/train.csv").set_index("text_id")
    featurizer = TermFrequencyFeaturizer()
    X = featurizer.featurize(data.full_text)
    y = data["vocabulary"]

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    model = catboost.CatBoostRegressor(iterations=2000, verbose=500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    error = mean_squared_error(y_pred, y_test) ** 0.5

    baseline_error = mean_squared_error([np.mean(y_train)] * len(y_test), y_test) ** 0.5
    print(f"Baseline error: {baseline_error}")
    print(f"Error: {error}")
