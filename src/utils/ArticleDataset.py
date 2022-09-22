from abc import ABC
from typing import Tuple, Iterator, List, Optional
import torch
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from src.datamodules.DocumentIterator import DocumentIterator
from src.utils.Preprocessor import Preprocessor
from src.utils.structures import Vocabularies


class ArticleDataset(IterableDataset, ABC):
    def __init__(self, vocabularies: Vocabularies, data_indexes: List[int],
                 data_path: str, doc_limit: Optional[int], file_extension: str, data_format: str):
        super().__init__()
        self.vocabularies: Vocabularies = vocabularies
        self.text_tensor_total_len: int = 512
        self.data_indexes = data_indexes

        self.data_path: str = data_path
        self.doc_limit: Optional[str] = doc_limit
        self.file_extension: str = file_extension
        self.data_format: str = data_format

    @staticmethod
    def pad(text_tensor, total):
        n = total - len(text_tensor)
        return F.pad(text_tensor, (0, n))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        di = 0
        i = 0
        iterator: DocumentIterator = DocumentIterator(data_path=self.data_path,
                                                      doc_limit=self.doc_limit,
                                                      file_extension=self.file_extension,
                                                      data_format=self.data_format)
        preprocessor: Preprocessor = Preprocessor(vocabularies=self.vocabularies)
        for article in iterator:
            title_integers: List[int] = preprocessor.preprocess_text(article.title)
            abstract_integers: List[int] = preprocessor.preprocess_text(article.abstract)
            mesh_integers: List[int] = preprocessor.preprocess_mesh(article.mesh_descriptors)
            journal_index = preprocessor.preprocess_journal(article.journal)
            if di >= len(self.data_indexes):
                # all articles from this dataset have been processed
                return
            if i != self.data_indexes[di]:
                # this element does not belong to the set
                i += 1
                continue
            i += 1
            di += 1

            text_tensor = torch.tensor(abstract_integers[:self.text_tensor_total_len], dtype=torch.float)
            text_len = len(text_tensor)
            if text_len == 0:
                continue
            text_tensor = ArticleDataset.pad(text_tensor, self.text_tensor_total_len)
            # get mesh_uids
            mesh_descriptors = torch.zeros(self.vocabularies.mesh_vocab_size, dtype=torch.int64)
            for mh in mesh_integers:
                mesh_descriptors[mh - 1] = 1
            # get journal tensor
            journal_tensor = torch.tensor(journal_index, dtype=torch.long)
            input_length = torch.tensor(text_len, dtype=torch.int64)
            yield text_tensor, journal_tensor, input_length, mesh_descriptors
