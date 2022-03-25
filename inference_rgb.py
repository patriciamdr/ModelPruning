import glob
import os
import re

import torch
from torch.utils.data import DataLoader

import config
from models.autoencoder import Autoencoder
from models.autoencoder_dataset import RGBDepthDataset
from models.pruned_model import PrunedModel

"""
Code from Task 3.
Inference method for all autoencoders before pruning the parallel branches.
"""
if __name__ == '__main__':
    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'lr': config.lr,
        'type': 'rgb',
        'train_original': True,
        'map_location': "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    checkpoint = os.path.join(config.ckpts_dir, 'Autoencoder/original_epoch=437-val_loss=1.40.ckpt')
    pretrained_ckpt = torch.load(checkpoint, map_location=config.device)
    pretrained_state_dict = pretrained_ckpt['state_dict']
    pretrained_model = Autoencoder(hparams=hparams)
    pretrained_model.load_state_dict(pretrained_state_dict)

    list_of_files = glob.glob(os.path.join(config.ckpts_dir, 'Autoencoder/Step/pruned-train-original/*.ckpt'))
    for i, path in enumerate(list_of_files):
        print(i)
        print(path)
        alpha = re.search('=(.+?)_', path)
        if alpha:
            alpha = alpha.group(1)
        print("Alpha: " + alpha)
        hparams['alpha'] = float(alpha)

        test_model = PrunedModel(hparams=hparams, pretrained_model=pretrained_model)
        checkpoint = torch.load(path, map_location=config.device)
        test_model.load_state_dict(checkpoint['state_dict'])
        test_model.eval()
        test_model.freeze()

        test_dataset = RGBDepthDataset(mode='test')
        test_dataloader = DataLoader(test_dataset)

        test_model.test_autoencoder(test_dataloader, float(alpha))

