"""
Test the performance of a pruned model compared to its original.
Comparison is done based on CPU/GPU memory consumption and time taken per test batch.
"""
import argparse
import glob
import os
import pandas as pd
import torch
import pytorch_lightning as pl

import config
from models.autoencoder import Autoencoder
from models.model_enum import Model
from models.vgg import VGGNet
from utils import GpuUsageLogger, CpuUsageLogger


def print_avg_performance(csv_path, use_gpu):
    """
    Compute and print average performance
    :param csv_path: Directory where measured data is located
    :param use_gpu: Specifies which file to evaluate
    """
    csv_files = sorted(glob.glob(os.path.join(csv_path, '*.csv'), recursive=True))

    for csv_file in csv_files:
        print(csv_file)
        if 'gpu' in csv_file and use_gpu:
            names = ['batch', 'time_taken', 'memory.used', 'memory.total']
            data = pd.read_csv(csv_file, names=names)
            avg_percentage_used = data['memory.used'].mean()
        elif 'cpu' in csv_file and not use_gpu:
            names = ['batch', 'time_taken', 'vsm']
            data = pd.read_csv(csv_file, names=names)
            avg_percentage_used = data['vsm'].mean()
        else:
            continue

        avg_time_taken = data['time_taken'].mean()
        print("Average time taken: " + str(avg_time_taken))
        print("Average unit consumption: " + str(avg_percentage_used))


def compare(original_model, smaller_model, csv_path, use_gpu):
    """
    Test performance of original and pruned model
    :param original_model: Original model used as pretrained model during pruning
    :param smaller_model: Final pruned model
    :param csv_path: Directory where to safe the output data
    :param use_gpu: Whether to test the performance on the GPU oder CPU
    :return:
    """
    csv_path_original = os.path.join(csv_path, 'original')
    csv_path_smaller = os.path.join(csv_path, 'smaller')

    if use_gpu:
        callback = GpuUsageLogger(csv_path_smaller)
    else:
        callback = CpuUsageLogger(csv_path_smaller)

    smaller_trainer = pl.Trainer(
        gpus=int(use_gpu),
        callbacks=[callback]
    )
    print("Test smaller model")
    smaller_trainer.test(smaller_model)
    print_avg_performance(csv_path_smaller, use_gpu)

    if use_gpu:
        callback = GpuUsageLogger(csv_path_original)
    else:
        callback = CpuUsageLogger(csv_path_original)

    original_trainer = pl.Trainer(
        gpus=int(use_gpu),
        callbacks=[callback]
    )
    print("Test original model")
    original_trainer.test(original_model)
    print_avg_performance(csv_path_original, use_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='Name of models',
                        default='VGG')
    parser.add_argument('--smaller_model_path',
                        default=os.path.join(config.ckpts_dir, 'VGG/Step/pruned-train-original/smaller_model.ckpt'),
                        help='Path to pruned model')
    parser.add_argument('--original_model_path',
                        default=os.path.join(config.ckpts_dir, 'VGG/original_epoch=149-val_loss=0.34.ckpt'),
                        help='Path to original model')
    parser.add_argument('--train_original', dest='train_original', action='store_true',
                        help='Whether to train the pretrained model or not')
    parser.set_defaults(prune_decoder=False)
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    args = parser.parse_args()

    map_location = "cpu"
    if args.use_gpu:
        if torch.cuda.is_available():
            map_location = "cuda:0"
        else:
            "CUDA not available, use CPU insted"
    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'train_original': args.train_original,
        'lr': config.lr,
        'map_location': map_location
    }

    # Load models
    if args.model is Model.VGG.value:
        smaller_checkpoint = torch.load(args.smaller_model_path, map_location=torch.device(hparams['map_location']))
        smaller_model = VGGNet(hparams=hparams, model_params=config.compute_pruned_vgg_params(config.remove_ratio))
        smaller_model.load_state_dict(smaller_checkpoint)

        original_checkpoint = torch.load(args.original_model_path, map_location=torch.device(hparams['map_location']))
        original_model = VGGNet(hparams=hparams)
        original_model.load_state_dict(original_checkpoint['state_dict'])
    elif args.model is Model.Autoencoder.value:
        hparams['type'] = 'rgb'
        hparams['prune_encoder'] = True
        hparams['prune_decoder'] = True
        smaller_checkpoint = torch.load(args.smaller_model_path, map_location=torch.device(hparams['map_location']))
        smaller_model = Autoencoder(hparams=hparams, model_params=config.compute_pruned_autoencoder_params(config.remove_ratio, True, True))
        smaller_model.load_state_dict(smaller_checkpoint)

        original_checkpoint = torch.load(args.original_model_path, map_location=torch.device(hparams['map_location']))
        original_model = Autoencoder(hparams=hparams)
        original_model.load_state_dict(original_checkpoint['state_dict'])

    csv_path = os.path.join(config.data_dir, args.model)
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    compare(original_model, smaller_model, csv_path, args.use_gpu)
