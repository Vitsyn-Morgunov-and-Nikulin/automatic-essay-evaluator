import pandas as pd

from src.text_cleaning.text_feature_extractor import TextFeatureExtractor


def test_text_feature_generation():
    train_csv = pd.read_csv("src/tests/data/train_sample.csv").full_text

    feature_extractor = TextFeatureExtractor()

    feature_df = feature_extractor.extract_features(train_csv)

    assert len(feature_df.columns) == 42
    assert len(feature_df) == 5
