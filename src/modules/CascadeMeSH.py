from typing import Dict, Optional
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from src.utils.MeSHTree import MeSHTree


class CascadeMeSH(pl.LightningModule):
    def __init__(self, mesh_tree: MeSHTree, lr: float, input_length: int, mesh_vocab_size: int):
        super().__init__()
        self.mesh_tree = mesh_tree
        self.classifiers = nn.ModuleDict()
        for node in self.mesh_tree.iter_without_leafs():
            self.classifiers[node.norm_tree_number] = nn.Linear(input_length, len(node.children_nodes))
        self.descriptor_linear_classifiers = nn.ModuleList(
            [nn.Linear(len(tree_numbers), 1) for tree_numbers in self.mesh_tree.tree_numbers_list])
        self.lr = lr

        self.loss_fct = nn.MultiLabelSoftMarginLoss()
        self.train_f = torchmetrics.F1Score(num_classes=2, threshold=0.1, average="micro",
                                            multiclass=True)
        self.val_f = torchmetrics.F1Score(num_classes=2, threshold=0.1, average="micro",
                                          multiclass=True)
        self.test_f = torchmetrics.F1Score(num_classes=2, threshold=0.1, average="micro",
                                           multiclass=True)
        self.save_hyperparameters()

    def forward(self, x):
        clf_result_dict: Dict[str, torch.tensor] = {}
        for node in self.mesh_tree.iter_without_leafs():
            clf = self.classifiers[node.norm_tree_number]
            result = torch.sigmoid(clf(x))
            # print('------------------------')
            # print(result)
            for child, value in zip(node.children_nodes, result.squeeze(0)):
                clf_result_dict[child.norm_tree_number] = value
                # print(value)
        logits = None
        for tree_numbers, linear in zip(self.mesh_tree.tree_numbers_list, self.descriptor_linear_classifiers):
            input_array = [clf_result_dict[tree_number.replace('.', '_')] if tree_number.replace('.', '_')
                                                                             in clf_result_dict else 0.0
                           for tree_number in tree_numbers]
            inputs = torch.tensor(input_array, device=self.device)
            # print('------------------')
            # print(inputs)
            logits = torch.cat((logits, linear(inputs))) if logits is not None else linear(inputs)
            # print(logits)
        return torch.nn.functional.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred = self.forward(x)
        loss = self.loss_fct(pred, y_mhs)
        self.log("train_loss", loss)
        self.train_f(pred, y_mhs)
        self.log("train_f", self.train_f, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred = self.forward(x)
        loss = self.loss_fct(pred, y_mhs)
        print('--------------------')
        print(pred)
        print(y_mhs)
        self.val_f(pred, y_mhs)
        self.log("val_f", self.val_f, on_step=True, on_epoch=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred = self.forward(x)
        loss = self.loss_fct(pred, y_mhs)
        self.test_f(pred, y_mhs)
        self.log("test_f", self.test_f, on_step=True, on_epoch=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
