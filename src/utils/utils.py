import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodules.DataModule import DataModule
from src.modules.CascadeMeSH import CascadeMeSH
from src.modules.MeSHProbeNet import MeSHProbeNet
from src.modules.SpikeMeSH import SpikeMeSH
from src.modules.SpikeText import SpikeText
from src.utils.MeSHTree import MeSHTree


def get_data_module(args) -> DataModule:
    return DataModule(data_path=args.data_path,
                      batch_size=args.batch_size,
                      train_frac=args.train_frac,
                      test_frac=args.test_frac,
                      doc_limit=args.doc_limit,
                      data_format=args.data_format,
                      num_workers=args.num_workers,
                      text_tensor_total_len=args.text_tensor_total_len,
                      saved_vocabularies_path=args.saved_vocabularies_path,
                      vocabularies_save_path=args.vocabularies_save_path)


def get_model(args, data_module: DataModule) -> pl.LightningModule:
    if args.checkpoint_path:
        if args.model == 'meshprobenet':
            ModelClass = MeSHProbeNet
        elif args.model == 'spiketext':
            ModelClass = SpikeText
        else:
            ModelClass = CascadeMeSH
        return ModelClass.load_from_checkpoint(args.ckpt_path)
    elif args.model == 'meshprobenet':
        return MeSHProbeNet(vocab_size=data_module.word_vocab_size, embed_dim=args.probenet_embed_dim, hidden_size=args.probenet_hidden_size,
                            n_layers=args.probenet_n_layers, n_probes=args.probenet_n_probes, n_jrnl=data_module.journal_vocab_size,
                            jrnl_dim=args.probenet_jrnl_dim, mesh_size=data_module.mesh_vocab_size, n_gpu=args.num_gpus,
                            batch_size=args.batch_size, lr=args.lr, weight_decay=args.probenet_weight_decay)
    elif args.model == 'spiketext':
        return SpikeText(num_inputs=args.num_inputs, num_hidden=args.num_hidden, beta=args.beta,
                         num_outputs=args.num_outputs, learning_rate=args.lr)
    elif args.model == 'cascademesh':
        mesh_tree = MeSHTree.build_original_mesh_tree(args.cascade_mesh_xml_path, mesh_vocabulary=data_module.vocabularies.mesh_vocabulary)
        return CascadeMeSH(mesh_tree=mesh_tree,
                           input_length=args.cascade_input_length,
                           lr=args.lr, mesh_vocab_size=data_module.mesh_vocab_size)
    else:
        raise AttributeError('Model selection value is inappropriate')


def get_model_checkpoint(args) -> ModelCheckpoint:
    return ModelCheckpoint(
        monitor=args.model_checkpoint_monitor,
        dirpath=args.checkpoint_save_path,
        filename=args.checkpoint_save_filename,
        save_top_k=args.save_top_k_checkpoints,
        every_n_train_steps=args.val_check_interval
    )
