import pandas as pd

from src.data_reader import load_train_test_df
from src.feature_extractors.term_frequency_extractor import TermFrequencyFeatureExtractor


def test_text_feature_generation():
    N_BINS = 20
    train_df, __ = load_train_test_df(is_testing=True)

    feature_extractor = TermFrequencyFeatureExtractor(n_bins=N_BINS)
    feature_df: pd.DataFrame = feature_extractor.generate_features(train_df.full_text)

    assert len(feature_df.columns) == N_BINS - 1
    assert len(feature_df) == 5
    assert not feature_df.isna().values.any()
