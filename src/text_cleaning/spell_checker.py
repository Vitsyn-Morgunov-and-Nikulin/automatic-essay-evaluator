__all__ = ['SmartSpellChecker']

import re
from collections import Counter
from functools import cache
from typing import Iterable

from spellchecker import SpellChecker


def get_word_counter(text: str):
    # removes punctuation and count words in sentence
    text = re.sub(r"[.,!?;:]", " ", text)

    return Counter(text.split())


custom_mappings = {
    'alot': 'a lot',
    'classwork': 'class work',
    'everytime': 'every time',
    'loosing': 'losing',
    'clases': 'classes',
    'payed': 'paid',
    'learnd': 'learned',
    'ect': 'etc',
    'wasnt': "wasn't",
    'wich': 'which',
    "sol's": 'souls',
    'thigs': 'things',
    'activies': 'activities',
    'oline': 'online',
    'thru': 'through',
    'inconclusion': 'in conclusion',
}

skipped_mappings = {
    ' u ': ' you ',
    'youll': "you will",
    'wont': "won't"}

exclude_words_from_check = {
    "you're", 'covid'
}

black_list = {'ther', "waldo's", "f's", ""}


class SmartSpellChecker:
    def __init__(self):
        self.spellcheck = SpellChecker()

    @cache
    def correct_word(self, mismatch: str):
        if mismatch in custom_mappings:
            return custom_mappings[mismatch]

        if mismatch in black_list:
            return ""

        if mismatch in exclude_words_from_check:
            return None

        # sometimes spellcheck thinks 'b' or 'c' if misspelled words
        # this condition > 2 is needed
        if len(mismatch) <= 2:
            return None

        return self.spellcheck.correction(mismatch)

    def correct_text(self, text: str):
        for key, value in skipped_mappings.items():
            if key in text:
                text = text.replace(key, value)

        word_count = get_word_counter(text)

        unknown_words = self.unknown(word_count)
        for misspell in unknown_words:
            correct = self.correct_word(misspell)
            if correct is not None:
                text = text.replace(misspell, correct)

        return text

    def unknown(self, words: Iterable):
        return self.spellcheck.unknown(words)
