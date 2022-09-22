import nltk
from typing import Dict, List, Optional, Tuple
import os
from indexingcode.utils.Preprocessor import Preprocessor
import json
from tqdm import tqdm
import gensim.utils

nltk.download('all', quiet=True)

data_dir = '/Users/wojciechsitek/Local/Indexing/data/BioAsq2021'
file_extension = '.json'

preprocessor = Preprocessor(gensim.utils.tokenize, nltk.PorterStemmer())

files: List[str] = os.listdir(data_dir)
files: List[str] = list(filter(lambda x: file_extension in x, files))


# abstracts
class AbstractIterator:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __iter__(self):
        for article in tqdm(self.raw_data):
            yield preprocessor.preprocess_text(article['abstractText'])

import gensim
from functools import reduce

def file_to_data(filename):
    with open(os.path.join(data_dir, filename), encoding='ISO-8859-1') as f:
            return json.load(f)['articles']

print("Files to data")
raw_datas = list(map(file_to_data, files))
print("Reduce")
full_raw_data =  reduce(lambda x, y: x + y, raw_datas)

model = gensim.models.Word2Vec(vector_size=1, window=5, min_count=2, epochs=2, sg=1)
sentences_iterator = AbstractIterator(full_raw_data)
print("Build vocabulary")
model.build_vocab(sentences_iterator)
print("Train")
model.train(sentences_iterator, total_examples=model.corpus_count, epochs=model.epochs)
print("Build vocab dictionaries")
num_to_emb = dict(map(lambda i: (i, model.wv[i][0]), preprocessor.vocabulary.values()))
num_to_text = dict(map(lambda a: (a[1], a[0]), preprocessor.vocabulary.items()))
print("Build abstracts out")
abstracts_out = list(map(lambda article: list(map(lambda i: num_to_emb[i], preprocessor.preprocess_text(article['abstractText']))), tqdm(full_raw_data)))

journals_out = list(map(lambda article: preprocessor.preprocess_journal(article['journal']), tqdm(full_raw_data)))
meshes_out = list(map(lambda article: preprocessor.preprocess_mesh(article['meshMajor']), tqdm(full_raw_data)))

out_dir = '/Users/wojciechsitek/Local/Indexing/out/article-preprocess-2021.nosync/'
with open(os.path.join(out_dir, 'num_to_emb.json'), 'w') as f:
    json.dump([(str(k),str(v)) for k, v in num_to_emb.items()], f)

with open(os.path.join(out_dir, 'num_to_text.json'), 'w') as f:
    json.dump([(str(k),str(v)) for k, v in num_to_text.items()], f)

with open(os.path.join(out_dir,'bioasq_preprocessed_all.csv'), 'w') as f:
    for line, journal, meshes in tqdm(zip(abstracts_out, journals_out, meshes_out)):
        f.write(f"{str(journal)};{','.join(map(str, line))};{','.join(map(str, meshes))}\n")

with open(os.path.join(out_dir, 'mesh_vocabulary.json'), 'w') as f:
    json.dump([(str(k),str(v)) for k, v in preprocessor.mesh_vocabulary.items()], f)
