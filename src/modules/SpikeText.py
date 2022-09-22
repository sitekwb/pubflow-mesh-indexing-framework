import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import snntorch as snn
import torch.nn.functional as F


class SpikeText(pl.LightningModule):
    def __init__(self, num_inputs, num_hidden, beta, num_outputs, learning_rate):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        # Initialize hidden states at t=0
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.lr = learning_rate
        self.loss_fct = F.cross_entropy
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, self.mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, self.mem2)
        return spk2, mem2

    def training_step(self, batch, batch_idx):
        x, y = batch
        spk, mem = self.forward(x)
        loss = self.loss_fct(mem, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_accuracy(mem, y)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        spk, mem = self.forward(x)
        loss = self.loss_fct(mem, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_accuracy(mem, y)
        self.log("test_acc", self.test_accuracy)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
