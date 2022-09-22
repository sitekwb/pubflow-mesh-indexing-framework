import nltk
from typing import Dict, List, Optional, Tuple
import json
import os
from indexingcode.utils.Preprocessor import Preprocessor
import ijson
from tqdm import tqdm
import gensim.utils
import gensim

nltk.download('all', quiet=True)

# data_path = '/media/wojtek/SanDisk/BioAsq-test2/bioasq_test_10mln.json'
data_path = '/media/wojtek/SanDisk/BioAsq2021/allMeSH_2021.json'

preprocessor = Preprocessor(gensim.utils.tokenize, nltk.PorterStemmer())


# abstracts
class AbstractIterator:
    def __init__(self, f):
        self.f = f

    def __iter__(self):
        self.f.seek(0)
        i = 0
        for article in tqdm(ijson.items(self.f, 'articles.item'), total=15559157):
            yield gensim.models.doc2vec.TaggedDocument(preprocessor.preprocess_text(f"{article['title']} {article['abstractText']}"), [i])
            i += 1


out_dir = './out/artprep/' #'indexingcode/out/article-preprocess.nosync/'
with open(data_path, encoding='ISO-8859-1') as f:
    model = gensim.models.Doc2Vec(vector_size=50, window=5, epochs=2, dbow_words=0)
    sentences_iterator = AbstractIterator(f)
    print("Build vocab")
    model.build_vocab(sentences_iterator)
    print("Train")
    model.train(sentences_iterator, total_examples=model.corpus_count, epochs=model.epochs)
    print("Build vocab dictionaries")

    num_to_text = dict(map(lambda a: (a[1], a[0]), preprocessor.vocabulary.items()))
    print("Build abstracts out")


    def iter_abstracts_out():
        f.seek(0)
        for article in ijson.items(f, 'articles.item'):
            yield (model.infer_vector([str(w) for w in preprocessor.preprocess_text(f"{article['title']} {article['abstractText']}")]),
                   preprocessor.preprocess_journal(article['journal']),
                   preprocessor.preprocess_mesh(article['meshMajor']))


    with open(os.path.join(out_dir, 'vectors.csv'), 'w') as f_vectors:
        with open(os.path.join(out_dir, 'journals.csv'), 'w') as f_journals:
            with open(os.path.join(out_dir, 'meshes.csv'), 'w') as f_meshes:
                for line, journal, meshes in tqdm(iter_abstracts_out(), total=15559157):
                    f_vectors.write(f"{','.join(map(str, line))}\n")
                    f_journals.write(f"{str(journal)}\n")
                    f_meshes.write(f"{','.join(map(str, meshes))}\n")

    with open(os.path.join(out_dir, 'num_to_text.json'), 'w') as fw:
        json.dump([(str(k), str(v)) for k, v in num_to_text.items()], fw)

    with open(os.path.join(out_dir, 'mesh_vocabulary.json'), 'w') as fw:
        json.dump([(str(k), str(v)) for k, v in preprocessor.mesh_vocabulary.items()], fw)
