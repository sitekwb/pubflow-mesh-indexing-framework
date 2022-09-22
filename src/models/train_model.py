from src.setup import setup
from src.utils.ArgumentParser import parse_arguments

if __name__ == '__main__':
    data_module, trainer, model = setup(parse_arguments())

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)
