from src.data_reader import load_train_test_df
from src.feature_extractors.text_statistics_extractor import \
    HandcraftedTextFeatureExtractor
from src.spell_checker import SmartSpellChecker


def test_text_feature_generation():
    train_df, test_df = load_train_test_df(is_testing=True)

    spellcheck = SmartSpellChecker()

    feature_extractor = HandcraftedTextFeatureExtractor(spellcheck)
    feature_df = feature_extractor.generate_features(train_df.full_text)

    assert len(feature_df.columns) == 42
    assert len(feature_df) == 5
