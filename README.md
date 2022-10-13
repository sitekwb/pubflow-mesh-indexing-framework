# PubFlow framework

- **Author**: Wojciech Bronisław Sitek
- **Framework keywords**: MeSH indexing, PubMed, SpikeText, MeSHProbeNet
- **Within Master Thesis**: Machine learning methods of automatic semantic indexing of medical texts using MeSH thesaurus
- **Tutor**: prof. dr hab. inż. Henryk Rybiński

The PubFlow framework is a tool for developers and researchers to create, implement, test and share MeSH indexing solutions.
In the repository, there are several tools implemented, such as MeSHProbeNet and SpikeText.

## Perequisites

- Conda distribution

## How to install

- Download project: `git clone https://github.com/sitekwb/pubflow-mesh-indexing-framework`
- Set this repository as a working directory: `cd pubflow-mesh-indexing-framework`
- Add this repository to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:[current-path]` ([current-path] may be obtained by `pwd` command)
- Create conda environment: `conda create -n pubflow python=3.9`
- Activate conda environment: `conda activate pubflow`
- Install requirements: `pip install -r requirements.txt`
- It may be possible on your machine to enable CUDA GPU to run your computations on the graphical card.

## How to use

Command to start training: `python src/models/train_model.py [--help] [ARGS]`
