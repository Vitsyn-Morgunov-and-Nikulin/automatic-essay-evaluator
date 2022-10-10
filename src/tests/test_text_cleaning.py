from src.data_reader import load_train_test_df
from src.spell_checker import SmartSpellChecker
from src.text_preprocessings.spellcheck_preprocessing import \
    SpellcheckTextPreprocessor


def test_text_cleaning():
    train_df, test_df = load_train_test_df(is_testing=True)

    spellcheck = SmartSpellChecker()
    text_preprocessor = SpellcheckTextPreprocessor(spellcheck)

    cleaned_texts = text_preprocessor.preprocess_data(train_df.full_text)

    assert len(cleaned_texts) == 5
