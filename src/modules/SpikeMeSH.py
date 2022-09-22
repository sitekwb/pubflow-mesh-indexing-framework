import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import snntorch as snn

from src.modules.MeSHProbeNet import BiRNN


class SpikeMeSH(pl.LightningModule):

    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, n_gpu, alpha, beta):
        super(SpikeMeSH, self).__init__()
        self.birnn = BiRNN(vocab_size, embed_dim, hidden_size, n_layers, n_gpu)
        self.snn = snn.Synaptic(alpha=alpha, beta=beta)
        self.loss_fct = nn.MultiLabelSoftMarginLoss()

    def forward(self, input_variables, input_lengths):
        birnn_outputs, birnn_hidden = self.birnn.forward(input_variables, input_lengths)
        logits = self.snn.forward(birnn_outputs)
        p = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fct(p)
        return loss.view(-1)

    def training_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred_mhs = self.forward(x, input_lengths)
        loss = self.loss_fct(pred_mhs, y_mhs)
        self.log("train_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred_mhs}

    def validation_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred_mhs = self.forward(x, input_lengths)
        loss = self.loss_fct(pred_mhs, y_mhs)
        self.log("val_loss", loss)
        return {'loss': loss, 'pred': pred_mhs}

    def test_step(self, batch, batch_idx):
        x, jrnl, input_lengths, y_mhs = batch
        pred_mhs = self.forward(x, input_lengths)
        loss = self.loss_fct(pred_mhs, y_mhs)
        self.log("test_loss", loss)
        return {'loss': loss, 'pred': pred_mhs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)