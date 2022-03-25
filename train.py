"""
Train a new model from scratch
"""
import os

import argparse as argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import config
import utils
from models.autoencoder import Autoencoder
from models.model_enum import Model
from models.vgg import VGGNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='Name of the model to train',
                        default='Autoencoder')
    args = parser.parse_args()
    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'lr': config.lr,
        'map_location': "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(config.logs_dir, args.model), name='original')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.patience,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config.ckpts_dir, args.model, 'original_{epoch:02d}-{val_loss:.2f}'),
        monitor='val_loss',
        mode='min'
    )

    # Create model object to train
    if args.model is Model.VGG.value:
        model = VGGNet(hparams=hparams)
    elif args.model is Model.Autoencoder.value:
        hparams['type'] = 'rgb'
        model = Autoencoder(hparams=hparams)

    model.apply(utils.weights_init)
    model = model.to(torch.device(hparams['map_location']))
    utils.checkParams(model)

    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=1,
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
