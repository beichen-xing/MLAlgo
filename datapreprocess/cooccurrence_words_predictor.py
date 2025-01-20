import re
import random
from collections import defaultdict


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words


def build_cooccurrence_model(words):
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 1):
        cooccurrence[words[i]][words[i + 1]] += 1
    return cooccurrence


def predict_next_word(cooccurrence, seed_word):
    if seed_word not in cooccurrence:
        return None
    next_words = cooccurrence[seed_word]
    return max(next_words, key=next_words.get)


