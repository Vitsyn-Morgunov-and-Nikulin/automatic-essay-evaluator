import pandas as pd

from src.text_cleaning.text_feature_extractor import TextFeatureExtractor


def test_text_cleaning():
    train_csv = pd.read_csv("src/tests/data/train_sample.csv").full_text

    feature_extractor = TextFeatureExtractor()

    text_series = feature_extractor.preprocess_texts(train_csv)

    assert len(text_series) == 5
