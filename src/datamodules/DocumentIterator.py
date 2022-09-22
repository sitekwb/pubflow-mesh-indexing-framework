import json
import os
from typing import Optional, List, Iterator

import ijson

from src.utils.structures import DocumentMetadata


class DocumentIterator:

    def __init__(self, data_path: str, doc_limit: Optional[int], file_extension: str, data_format: str):
        self.data_path: str = data_path
        self.doc_limit: Optional[int] = doc_limit
        self.file_extension: str = file_extension
        self.filenames: List[str] = self.get_filenames()
        self.data_format: str = data_format
        self.dataset_size: int = 0

    def get_filenames(self) -> List[str]:
        return list(filter(lambda x: self.file_extension in x, os.listdir(self.data_path)))

    def iter_bioasq(self) -> Iterator[DocumentMetadata]:
        i = 0
        for filename in self.filenames:
            with open(os.path.join(self.data_path, filename), encoding='ISO-8859-1') as f:
                for article in ijson.items(f, "articles.item"):
                    i += 1
                    if self.doc_limit and i > self.doc_limit:
                        self.dataset_size = self.doc_limit
                        return
                    if not article['title'] or not article['abstractText'] or not article['journal'] or not article['meshMajor']:
                        continue
                    yield DocumentMetadata(
                        title=article['title'],
                        abstract=article['abstractText'],
                        journal=article['journal'],
                        mesh_descriptors=article['meshMajor']
                    )
        self.dataset_size = i
    #     python src/models/train_model.py meshprobenet --data-path /home/wojciechsitek/master-indexing/data --data-format=bioasq --vocabularies-save-path ~/master-indexing/vocabularies.json --model meshprobenet --batch-size 32 --doc-limit 10000 --checkpoint-save-path ~/master-indexing/lightning_logs --max-epochs 1 --seed 29

    def iter_medline(self) -> Iterator[DocumentMetadata]:
        pass
        # for filename in self.files:
        #     full_filename_path: str = os.path.join(self.data_module.data_dir, filename)
        #     tree = ET.parse(full_filename_path)
        #     root = tree.getroot()
        #     for article_node in tqdm(root.iter('MedlineCitation')):
        #         # preprocess
        #         abstract_text: str = get_text(article_node.find('Article/Abstract/AbstractText'))
        #         title_text: str = get_text(article_node.find('Article/ArticleTitle'))
        #         journal_name_text: str = get_text(article_node.find('Article/Journal/Title'))
        #
        #         processed_values = self.preprocess(abstract_text, title_text, journal_name_text)
        #         abstract_integers, abstract_tokens, title_integers, title_tokens, journal_index = processed_values
        #         # save to db
        #         try:
        #             self.data_module.save_article_to_db(article_node, abstract_integers, title_integers, journal_index)
        #         except:
        #             print(sys.exc_info())
        #         # return iteration element
        #         yield title_tokens + abstract_tokens
        # TODO iter medline

    def iter_pmc(self) -> Iterator[DocumentMetadata]:
        pass
        # for filename in self.files:
        #     full_filename_path: str = os.path.join(self.data_module.data_dir, filename)
        #     tree = ET.parse(full_filename_path)
        #     root = tree.getroot()
        #     for article_node in root.iter('article'):
        #         # preprocess
        #         abstract_text: str = ' '.join(
        #             [''.join(p.itertext()) for p in article_node.findall('front/article-meta/abstract//p')])
        #         title_text: str = get_text(article_node.find('front/article-meta/title-group/article-title'))
        #         journal_name_text: str = get_text(article_node.find('front/journal-meta/journal-title'))
        #
        #         processed_values = self.preprocess(abstract_text, title_text, journal_name_text)
        #         abstract_integers, abstract_tokens, title_integers, title_tokens, journal_index = processed_values
        #         self.data_module.save_article_to_db(article_node, abstract_integers, title_integers, journal_index)
        #         # return iteration element
        #         yield title_tokens + abstract_tokens
        # TODO iter pmc

    def __iter__(self) -> Iterator[DocumentMetadata]:
        if self.data_format == 'bioasq':
            return self.iter_bioasq()
        elif self.data_format == 'medline':
            return self.iter_medline()
        elif self.data_format == 'pmc':
            return self.iter_pmc()
        else:
            raise AttributeError("Unsupported data format")
