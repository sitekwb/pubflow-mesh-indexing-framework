import string
from typing import Dict, List, Optional, Union
import nltk
from nltk.corpus import stopwords
from collections import Counter
import gensim

from src.utils.structures import Vocabularies, DocumentMetadata


class Preprocessor:
    def __init__(self, testing=False, vocabularies: Vocabularies = None):
        self.tokenize = gensim.utils.tokenize
        self.stemmer: nltk.stem.PorterStemmer = nltk.stem.PorterStemmer()
        self.counter = Counter()
        self.current_index: int = 0
        nltk.download('stopwords', quiet=True)
        stop_words: List[str] = stopwords.words('english')
        punctuation: List[str] = list(string.punctuation)
        self.stopwords_and_punctuation: List[str] = stop_words + punctuation
        self.testing = testing
        self.num_to_text: Dict[int, str] = {}
        self.vocabularies = vocabularies if vocabularies else Vocabularies(word_vocabulary={},
                                                                           mesh_vocabulary={},
                                                                           journal_vocabulary={})

    def preprocess_journal(self, journal_name) -> int:
        index: Optional[int] = self.vocabularies.journal_vocabulary.get(journal_name)
        if self.testing:
            return index if index else 0
        if not index:
            index = self.vocabularies.journal_vocab_size
            self.vocabularies.journal_vocabulary[journal_name] = index
        return index

    def preprocess_text(self, text: str, only_tokenize=False) -> List[int]:
        tokens_categorical: List[int] = []
        for t in self.tokenize(text.lower()):
            if not only_tokenize:
                token: str = self.stemmer.stem(t)
                if token in self.stopwords_and_punctuation:
                    continue
            else:
                token = t

            # to categorical
            index: Optional[int] = self.vocabularies.word_vocabulary.get(token)
            if not index:
                if self.testing:
                    continue
                self.vocabularies.word_vocabulary[token] = self.current_index
                self.num_to_text[self.current_index] = token
                index = self.current_index
                self.current_index += 1

            tokens_categorical.append(index)
        return tokens_categorical

    def preprocess_mesh(self, mesh_descriptors: List[Union[str, int]]) -> List[int]:
        mesh_integers: List[int] = []
        for descriptor_name in mesh_descriptors:
            index: Optional[int] = self.vocabularies.mesh_vocabulary.get(descriptor_name)
            if not index:
                if self.testing:
                    continue
                index = self.vocabularies.mesh_vocab_size
                self.vocabularies.mesh_vocabulary[descriptor_name] = index
            mesh_integers.append(index)
        return mesh_integers

    def preprocess_metadata(self, article: DocumentMetadata) -> None:
        self.preprocess_text(article.title)
        self.preprocess_text(article.abstract)
        self.preprocess_mesh(article.mesh_descriptors)
        self.preprocess_journal(article.journal)
