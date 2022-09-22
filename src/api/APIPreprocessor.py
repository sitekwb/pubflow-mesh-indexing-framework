from typing import Optional, List
import torch
import torch.nn.functional as F
from nltk import PorterStemmer
import json

from src.utils.Preprocessor import Preprocessor


class APIPreprocessor:
    def __init__(self, saved_data_file, preprocessor=Preprocessor(tokenizer_name='spacy', stemmer=PorterStemmer())):
        self.preprocessor = preprocessor
        self.text_tensor_total_len = 1024
        self.load_preprocessor_hparams(saved_data_file)

    def load_preprocessor_hparams(self, saved_data_file):
        with open(saved_data_file) as file:
            saving_data = json.load(file)
            self.preprocessor.mesh_vocab_size_saved = saving_data['mesh_vocab_size']
            self.preprocessor.journal_vocab_size_saved = saving_data['journal_vocab_size']
            self.preprocessor.vocab_size_saved = saving_data['vocab_size']
            self.preprocessor.mesh_vocabulary = saving_data['mesh_vocabulary']
            self.preprocessor.journal_vocabulary = saving_data['journal_vocabulary']
            self.preprocessor.vocabulary = saving_data['vocabulary']

    @staticmethod
    def pad(text_tensor, total):
        n = total - len(text_tensor)
        return F.pad(text_tensor, (0, n))

    def get_input_data(self, abstract_text, title_text, journal_name_text, mesh_descriptors: List[str]):
        abstract_integers = self.preprocessor.preprocess_text(abstract_text)
        # title_integers = self.preprocessor.preprocess_text(title_text)
        journal_index = self.preprocessor.preprocess_journal(journal_name_text)
        mesh_integers = self.preprocessor.preprocess_mesh(mesh_descriptors)

        # TODO duplicate
        # text_tensor = torch.tensor(abstract_integers[:self.text_tensor_total_len], dtype=torch.long)
        # text_len = len(text_tensor)
        # text_tensor = APIPreprocessor.pad(text_tensor, self.text_tensor_total_len)
        # # get mesh_uids
        # mesh_descriptors = torch.zeros(self.preprocessor.mesh_vocab_size, dtype=torch.int64)
        # for mh in mesh_integers:
        #     mesh_descriptors[mh - 1] = 1
        # # get journal tensor
        # journal_tensor = torch.tensor(journal_index, dtype=torch.long)
        # input_length = torch.tensor(text_len, dtype=torch.int64)
        # return text_tensor, journal_tensor, input_length, mesh_descriptors
