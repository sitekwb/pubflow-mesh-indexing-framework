import argparse
import pytorch_lightning as pl
from src.utils.ArgumentParser import parse_arguments
from src.utils.utils import get_model, get_data_module, DataModule, get_model_checkpoint


def setup(args: argparse.Namespace):
    data_module: DataModule = get_data_module(args)
    data_module.prepare()
    model: pl.LightningModule = get_model(args, data_module)
    checkpoint_callback = get_model_checkpoint(args)

    trainer = pl.Trainer(default_root_dir=args.checkpoint_save_path,
                         max_epochs=args.max_epochs,
                         gpus=args.num_gpus,
                         callbacks=[checkpoint_callback],
                         val_check_interval=args.val_check_interval
                         )

    return data_module, trainer, model


if __name__ == '__main__':
    setup(parse_arguments())
