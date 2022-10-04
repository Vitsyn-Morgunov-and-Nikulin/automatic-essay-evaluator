import pandas as pd

from src.text_cleaning.spell_checker import SmartSpellChecker
from src.text_cleaning.text_preprocessing import TextPreprocessor


def test_text_cleaning():
    train_csv = pd.read_csv("src/tests/data/train_sample.csv").full_text

    spellcheck = SmartSpellChecker()
    text_preprocessor = TextPreprocessor(spellcheck)

    text_series = text_preprocessor.preprocess_texts(train_csv)

    assert len(text_series) == 5
