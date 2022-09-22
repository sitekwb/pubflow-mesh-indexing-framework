from typing import Optional, Dict, List

import pydantic


class Vocabularies(pydantic.BaseModel):
    dataset_size: Optional[int]
    word_vocabulary: Dict[str, int]
    mesh_vocabulary: Dict[str, int]
    journal_vocabulary: Dict[str, int]

    @property
    def word_vocab_size(self) -> int:
        return len(self.word_vocabulary) + 1

    @property
    def journal_vocab_size(self) -> int:
        return len(self.journal_vocabulary) + 1

    @property
    def mesh_vocab_size(self) -> int:
        return len(self.mesh_vocabulary)


class DocumentMetadata(pydantic.BaseModel):
    title: str
    abstract: str
    journal: str
    mesh_descriptors: List[str]


class BioasqJsonArticle(pydantic.BaseModel):
    title: str
    abstractText: str
    journal: str
    meshMajor: List[str]


class BioasqJson(pydantic.BaseModel):
    articles: List[BioasqJsonArticle]
