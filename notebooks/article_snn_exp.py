from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, data, classes):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float)
        self.classes = torch.tensor(classes, dtype=torch.int)

    def __getitem__(self, i):
        return self.data[i], self.classes[i]

    def __len__(self):
        return len(self.data)



class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str, train_classes_path: str, test_path: str, test_classes_path: str, batch_size: int = 1):
        super().__init__()
        self.train_path = train_path
        self.train_classes_path = train_classes_path
        self.test_path = test_path
        self.test_classes_path = test_classes_path

        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_classes = None
        self.test_classes = None
        self.sep = ';'
        self.num_classes = None
        self.batch_size = batch_size
        self.vector_len = None

    @staticmethod
    def pad(text_tensor, total):
        n = total - len(text_tensor)
        return F.pad(text_tensor, (0, n))

    def prepare_data(self):
        self.download_dataset()
        self.num_classes = len(set(self.train_classes))

    def download_dataset(self):
        with open(self.train_path) as f:
            self.train_data = [[float(_) for _ in line.split(self.sep)] for line in f]
        with open(self.test_path) as f:
            self.test_data = [[float(_) for _ in line.split(self.sep)] for line in f]
        with open(self.train_classes_path) as f:
            self.train_classes = [[int(line)] for line in f]
        with open(self.test_classes_path) as f:
            self.test_classes = [[int(line)] for line in f]

    def setup(self, stage: Optional[str] = None):
        self.vector_len = self.count_vector_len()
        self.train_data = [i + [0]*(self.vector_len-len(i)) for i in self.train_data]
        self.test_data = [i + [0]*(self.vector_len-len(i)) for i in self.test_data[:self.vector_len]]
        self.train_dataset = MyDataset(data=self.train_data, classes=self.train_classes)
        self.test_dataset = MyDataset(data=self.test_data, classes=self.test_classes)

    def count_vector_len(self):
        max_len = 0
        for vector in self.train_data:
            max_len = max(max_len, len(vector))
        return max_len

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    @property
    def num_inputs(self):
        return self.vector_len

    @property
    def num_outputs(self):
        return 1


from models.SpikeText import SpikeText
from pytorch_lightning.callbacks import ModelCheckpoint

data_module = MyDataModule(train_path='/data/DatasetsPM/TitleMeshIB/TrainTitleIB.csv',
                           test_path='/data/DatasetsPM/TitleMeshIB/TestTitleIB.csv',
                           train_classes_path='/data/DatasetsPM/TitleMeshIB/TrainTitleIBClass.csv',
                           test_classes_path='/data/DatasetsPM/TitleMeshIB/TestTitleIBClass.csv'
                           )
model = SpikeText(num_inputs=data_module.num_inputs, num_hidden=100, beta=0.95,
                  num_outputs=data_module.num_outputs, learning_rate=1e-4)


checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath='lightning_logs',
        filename="{epoch:02d}-{val_loss:.2f}",
    )

trainer = pl.Trainer(default_root_dir='../indexingcode/lightning_logs',
                     max_epochs=1, gpus=torch.cuda.device_count(),
                     callbacks=[checkpoint_callback])

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)