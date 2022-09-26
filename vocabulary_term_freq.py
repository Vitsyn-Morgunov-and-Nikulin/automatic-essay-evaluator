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

    def __init__(self):
        term_frequencies_dataset = self._load_term_frequency_dataset()
        self.term2freq: Dict[str, int] = defaultdict(lambda: 0)
        self.term2freq.update(term_frequencies_dataset.set_index("word").to_dict())
        nltk.download("punkt")

    def _load_term_frequency_dataset(self) -> pd.DataFrame:
        try:
            term_frequencies_dataset = pd.read_csv("aux_data/unigram_freq.csv")
        except IOError:
            raise IOError(
                "Guys, you need `unigram_freq.csv` dataset in aux_data/ folder"
            )
        return term_frequencies_dataset

    def featurize(self, texts: pd.Series, n_bins: int = 40) -> pd.DataFrame:
        term_frequencies = texts.apply(self._compute_term_frequencies_from_text)
        min_bin = 0
        max_bin = np.log1p(self.MAX_TERM_FREQ)
        bins = np.linspace(min_bin, max_bin, n_bins)
        feature_names = [
            f"bin_{round(bins[i], 1)}_{round(bins[i+1], 1)}"
            for i in range(len(bins) - 1)
        ]
        feature_values = []
        for i, word_frequencies in enumerate(tqdm.tqdm(term_frequencies.values)):
            word_frequencies_log = np.log1p(word_frequencies)
            term_frequencies_histogram_values, __ = np.histogram(
                word_frequencies_log, bins=bins
            )
            normalized_term_frequencies_histogram_values = (
                term_frequencies_histogram_values / len(word_frequencies)
            )
            feature_values.append(normalized_term_frequencies_histogram_values)
        feature_values = np.array(feature_values)
        feature_df = pd.DataFrame(
            feature_values, columns=feature_names, index=texts.index
        )
        return feature_df

    def _compute_term_frequencies_from_text(self, text: str) -> List[int]:
        tokens = nltk.tokenize.word_tokenize(text)
        words = [token.lower() for token in tokens if token.isalpha()]
        word_frequencies = [self.term2freq[word] for word in words]
        return word_frequencies


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
