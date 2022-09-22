import argparse
import os
import sys

import torch


def spiketext_add_arguments(parser):
    parser.add_argument('--spike-snn-hidden-size', type=int, default=1000, help='Size of hidden layer of SpikeText')
    parser.add_argument('--spike-snn-beta', type=float, default=0.95, help='Beta parameter of SpikeText')
    parser.add_argument('--spike-num-inputs', type=int, default=512, help='Size of input layer of SpikeText')
    parser.add_argument('--spike-num-outputs', type=int, default=16, help='Size of output layer of SpikeText')


def meshprobenet_add_arguments(parser):
    parser.add_argument('--probenet-embed-dim', type=int, default=250, help='Dimension of embedding')
    parser.add_argument('--probenet-hidden-size', type=int, default=200, help='Size of the hidden layer')
    parser.add_argument('--probenet-n-layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--probenet-n-probes', type=int, default=25, help='Number of probes')
    parser.add_argument('--probenet-jrnl-dim', type=int, default=100, help='Journal dimension')
    parser.add_argument('--probenet-weight-decay', type=float, default=0, help='L2 Regularization weight decay.')


def cascademesh_add_arguments(parser):
    parser.add_argument('--cascade-mesh-xml-path', type=str, default='data/raw/desc2018.xml',
                        help='A path to the XML of the MeSH thesaurus')
    parser.add_argument('--cascade-input-length', type=int, default=512, help='The input length of the CascadeMeSH')


def parse_arguments():
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument('--data-path', type=str, default='data/raw/bioasq-2018', help='Path to data directory')
    parser.add_argument('--data-format', type=str, choices=['pmc', 'medline', 'bioasq'], default='bioasq',
                        help='The choice of data format')
    parser.add_argument('--train-frac', type=float, default=0.895, help='Fraction of documents in a training set')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Fraction of documents in a test set')
    parser.add_argument('--saved-vocabularies-path', type=str, default=None,
                        help='Path to a saved vocabularies JSON file')
    parser.add_argument('--vocabularies-save-path', type=str, default=None,  # data/processed/vocabularies.json
                        help='Path for saving processed vocabularies in a JSON file')

    # MODEL
    parser.add_argument('--model', type=str, default='meshprobenet', choices=['meshprobenet', 'spiketext', 'cascademesh'],
                        help='Source model')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--text-tensor-total-len', type=int, default=512, help='A total length of the text tensor.')

    # TECHNICAL DETAILS
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of workers')
    parser.add_argument('--doc-limit', type=int, default=None, help='A limit of documents to train and test on')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint file, on which training will continue')
    parser.add_argument('--checkpoint-save-path', type=str, default='out/checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--num-gpus', type=int, default=int(torch.cuda.device_count()), help='Number of GPU cores')
    parser.add_argument('--max-epochs', type=int, default=5, help='Maximum number of training epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility.')
    parser.add_argument('--model-checkpoint-monitor', type=str, default="val_f_epoch", help='Checkpoint monitor value.')
    parser.add_argument('--checkpoint-save-filename', type=str, default="{epoch:02d}-{val_f_epoch:.2f}",
                        help='Checkpoint save filename format.')
    parser.add_argument('--val-check-interval', type=int, default=None, help='Validation check interval')
    parser.add_argument('--save-top-k-checkpoints', type=int, default=10, help='Save top k checkpoints')

    meshprobenet_add_arguments(parser)
    spiketext_add_arguments(parser)
    cascademesh_add_arguments(parser)

    return parser.parse_args()
