from collections import defaultdict
import re


class SearchIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.documents = {}

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        words = set(re.findall(r'\w+', text.lower()))
        for word in words:
            self.index[word].add(doc_id)

    def search_single_word(self, word):
        return self.index.get(word.lower(), set())

    def search_multiple_words(self, words):
        word_set = set(words)
        matching_docs = None
        for word in word_set:
            word_docs = self.index.get(word.lower(), set())
            if matching_docs is None:
                matching_docs = word_docs
            else:
                matching_docs &= word_docs
        return matching_docs

    def search_sentence(self, sentence):
        words = re.findall(r'\w+', sentence.lower())
        matching_docs = set(self.search_multiple_words(words))
        ordered_matches = set()
        for doc_id in matching_docs:
            doc_text = self.documents[doc_id].lower()
            if self._check_ordered_match(doc_text, words):
                ordered_matches.add(doc_id)
        return ordered_matches

    def _check_ordered_match(self, text, words):
        pattern = r'.*?'.join(map(re.escape, words))
        return re.search(pattern, text) is not None


index = SearchIndex()
index.add_document(1, "The quick brown fox jumps over the lazy dog")
index.add_document(2, "A fast brown animal jumps over a sleepy dog")

print(index.search_single_word("brown"))
print(index.search_multiple_words(["fox", "dog"]))
print(index.search_sentence("brown fox jumps"))