"""This file contains functions that can generates hand-crafted features from the text"""

__all__ = [
    'count_words',
    'count_punctuation',
    'count_how_many_words_are_repeating',
    'count_misspelled_words',
    'preprocess_test'
]

import re
from collections import Counter
from string import punctuation
from typing import Dict

from src.text_cleaning.spell_checker import SmartSpellChecker

underscores_to_replace = {
    'Generic_Name': 'name',
    'OTHER_NAME': 'name',
    'STUDENT_NAME': 'name',
    'Generic_Namea': 'name',
    'PROPER_NAME': 'proper name',
    'PROEPR_NAME': 'proper name ',
    'Generic_School': 'school',
    'SCHOOL_NAME': 'school',
    'Generic_school': 'school',
    'TEACHER_NAME': 'teacher',
    'Generic_City': 'city',
    'LOCATION_NAME': 'location',
    'STORE_NAME': 'store',
    'RESTAURANT_NAME': 'restaurant',
    'LANGUAGE_NAME': 'language',
}


def preprocess_test(text: str) -> str:
    # Removes digits, special signs, double spaces and tabulation, underscores
    for key, value in underscores_to_replace.items():
        text = text.replace(key, value)

    text = re.sub(r"[\d%@\\#$&^\"_()*+\-/]", " ", text)
    text = re.sub(r"\n|\t", " ", text)
    text = re.sub(r'(?<=[.,:;!?])(?=\S)', " ", text)  # Add space after punctuation
    text = re.sub(r"\s+", " ", text)

    return text


def get_word_counter(text: str) -> Dict[str, int]:
    # removes punctuation and count words in sentence
    text = re.sub(r"[.,!?;:]", " ", text)

    return Counter(text.split())


def count_punctuation(text: str) -> Dict[str, int]:
    features = {}

    for symbol in (punctuation + " "):
        features[f'count_{symbol}'] = text.count(symbol)

    return features


def count_how_many_words_are_repeating(text: str) -> Dict[str, int]:
    word_count = get_word_counter(text)
    features = {}

    # For each text count how many unique words repeated >= times
    for ii in range(3, 10):
        n_words_repeated = len([word for word in word_count if word_count[word] >= ii])
        features[f'{ii}_word_repeated'] = n_words_repeated

    return features


def count_misspelled_words(text: str, spellcheck: SmartSpellChecker) -> Dict[str, int]:
    unknown_words = spellcheck.unknown(get_word_counter(text))
    return {'n_misspelled_words': len(unknown_words)}


def count_words(text: str) -> Dict[str, int]:
    return {'length': len(get_word_counter(text))}
