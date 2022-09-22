import json
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodules.DocumentIterator import DocumentIterator
from src.utils.ArticleDataset import ArticleDataset
from src.utils.Preprocessor import Preprocessor
from src.utils.structures import Vocabularies


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, train_frac: float, test_frac: float,
                 num_workers: int, doc_limit: Optional[int],
                 data_format: str, saved_vocabularies_path: Optional[str],
                 vocabularies_save_path: Optional[str], text_tensor_total_len: int):
        super().__init__()
        self.data_path: str = data_path
        self.batch_size: int = batch_size
        self.num_workers = num_workers
        self.train_frac: float = train_frac
        self.test_frac: float = test_frac
        self.doc_limit: Optional[int] = doc_limit
        self.data_format: str = data_format
        self.saved_vocabularies_path: Optional[str] = saved_vocabularies_path
        self.vocabularies_save_path: Optional[str] = vocabularies_save_path
        self.text_tensor_total_len: int = text_tensor_total_len

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_data_indexes = None
        self.test_data_indexes = None
        self.val_data_indexes = None

        self.vocabularies = Vocabularies(word_vocabulary={},
                                         mesh_vocabulary={},
                                         journal_vocabulary={})
        self.file_extension = {
            'bioasq': '.json',
            'medline': '.xml',
            'pmc': '.xml'
        }[self.data_format]

    def make_vocabularies(self):
        preprocessor: Preprocessor = Preprocessor()
        iterator: DocumentIterator = DocumentIterator(data_path=self.data_path,
                                                      doc_limit=self.doc_limit,
                                                      file_extension=self.file_extension,
                                                      data_format=self.data_format)
        dataset_size = 0
        for article in tqdm(iterator):
            preprocessor.preprocess_metadata(article)
            dataset_size += 1
        self.vocabularies = preprocessor.vocabularies
        self.vocabularies.dataset_size = dataset_size

        if self.vocabularies_save_path:
            with open(self.vocabularies_save_path, 'w+') as f:
                json.dump(self.vocabularies.dict(), f)

    def fetch_vocabularies(self):
        with open(self.saved_vocabularies_path) as file:
            self.vocabularies = Vocabularies(**json.load(file))

    def setup(self, stage: Optional[str] = None):
        train_size: int = int(self.train_frac * self.vocabularies.dataset_size)
        test_size: int = int(self.test_frac * self.vocabularies.dataset_size)

        permutation = np.random.permutation(np.arange(self.vocabularies.dataset_size))
        if self.doc_limit:
            permutation = permutation[:self.doc_limit]
        self.train_data_indexes, remaining_data_indexes = np.split(permutation, [train_size])
        self.test_data_indexes, self.val_data_indexes = np.split(remaining_data_indexes, [test_size])
        self.train_data_indexes = sorted(self.train_data_indexes)
        self.test_data_indexes = sorted(self.test_data_indexes)
        self.val_data_indexes = sorted(self.val_data_indexes)
        self.train_dataset = ArticleDataset(vocabularies=self.vocabularies,
                                            data_indexes=self.train_data_indexes,
                                            data_path=self.data_path,
                                            doc_limit=self.doc_limit,
                                            file_extension=self.file_extension,
                                            data_format=self.data_format
                                            )
        self.test_dataset = ArticleDataset(vocabularies=self.vocabularies,
                                           data_indexes=self.test_data_indexes,
                                           data_path=self.data_path,
                                           doc_limit=self.doc_limit,
                                           file_extension=self.file_extension,
                                           data_format=self.data_format
                                           )
        self.val_dataset = ArticleDataset(vocabularies=self.vocabularies,
                                          data_indexes=self.val_data_indexes,
                                          data_path=self.data_path,
                                          doc_limit=self.doc_limit,
                                          file_extension=self.file_extension,
                                          data_format=self.data_format
                                          )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def word_vocab_size(self):
        return self.vocabularies.word_vocab_size

    @property
    def journal_vocab_size(self):
        return self.vocabularies.journal_vocab_size

    @property
    def mesh_vocab_size(self):
        return self.vocabularies.mesh_vocab_size

    def prepare(self):
        if self.saved_vocabularies_path:
            self.fetch_vocabularies()
        else:
            self.make_vocabularies()
