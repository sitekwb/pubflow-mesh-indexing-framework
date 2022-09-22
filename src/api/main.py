from typing import Optional, List

from fastapi import FastAPI

from MeSHProbeNet import MeSHProbeNet
from APIPreprocessor import APIPreprocessor

model = MeSHProbeNet.load_from_checkpoint(
    'C:\\Users\\swojc\\IdeaProjects\\master-indexing\\code\\indexing-code\\lightning_logs\\meshprobenet-epoch=00-val_f_epoch=0.12.ckpt')
api_preprocessor = APIPreprocessor(
    saved_data_file='C:\\Users\\swojc\\IdeaProjects\\master-indexing\\code\\indexing-code\\out\\bioasq\\data-test2.json')


app = FastAPI()

@app.post("/predict")
async def read_root(abstract: str, title: str, journal_name: str, mesh_desc_names: List[str]):
    text, journal, length, mesh = api_preprocessor.get_input_data(abstract, title, journal_name, mesh_desc_names)
    return model.forward(text, length, journal)
