"""
Utils file consisting of static methods used for training, pruning and testing.
"""
import os
import shutil
import subprocess
import time
from typing import List, Tuple, Dict

import cv2
import psutil
import numpy as np
import pandas as pd
import torch
import torchvision

from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import random_split
from torchvision import transforms

import config
from models.autoencoder_dataset import RGBDepthDataset


def weights_init(net):
    """
    Initialize model weights
    """
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal(m.weight, mode='fan_out')
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal(m.weight, std=1e-3)


class CpuUsageLogger(Callback):
    """
    Callback to measure RAM usage and duration per test batch. Output is saved in a CSV file.
    """
    def __init__(self, csv_path):
        self.batch = 0
        self.csv_path = csv_path
        self.p = psutil.Process(os.getpid())

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # print("On test batch start")
        self.start = time.time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # print("On test batch end")
        time_taken = time.time() - self.start
        new_data = pd.DataFrame()
        df = pd.DataFrame([{'batch': self.batch,
                            'time_taken': time_taken,
                            'vms': self.p.memory_info().vms}])
        new_data = new_data.append(df, ignore_index=True)

        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        new_data.to_csv(os.path.join(self.csv_path, 'cpu_usage_and_time_taken.csv'), index=False, mode='a+',
                        header=False)
        self.batch += 1


# Modified version of
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/gpu_stats_monitor.py
class GpuUsageLogger(Callback):
    """
    Callback to measure GPU memory consumption and duration per test batch. Output is saved in a CSV file.
    """
    def __init__(self, csv_path):
        super(GpuUsageLogger, self).__init__()
        self.batch = 0
        self.csv_path = csv_path

        if shutil.which('nvidia-smi') is None:
            raise MisconfigurationException(
                'Cannot use GPUStatsMonitor callback because NVIDIA driver is not installed.'
            )

    def on_test_start(self, trainer, pl_module):
        if not trainer.on_gpu:
            raise MisconfigurationException(
                'You are using GPUStatsMonitor but are not running on GPU'
                f' since gpus attribute in Trainer is set to {trainer.gpus}.'
            )

        self._gpu_ids = ','.join(map(str, trainer.data_parallel_device_ids))

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.start = time.time()

        """Get the GPU stats keys"""
        gpu_stat_keys = [('memory.used', 'MB'), ('memory.total', 'MB')]
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        time_taken = time.time() - self.start

        """Get the GPU stats keys"""
        gpu_stat_keys = [('utilization.gpu', '%')]
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)

        new_data = pd.DataFrame()
        data = dict({'batch': self.batch, 'time_taken': time_taken})
        data.update(logs)
        df = pd.DataFrame([data])
        new_data = new_data.append(df, ignore_index=True)

        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        new_data.to_csv(os.path.join(self.csv_path, 'gpu_usage_and_time_taken.csv'), index=False, mode='a+',
                        header=False)
        self.batch += 1

    @staticmethod
    def _parse_gpu_stats(gpu_ids: str, stats: List[List[float]], keys: List[Tuple[str, str]]) -> Dict[str, float]:
        """Parse the gpu stats into a loggable dict"""
        logs = {}
        for i, gpu_id in enumerate(gpu_ids.split(',')):
            for j, (x, unit) in enumerate(keys):
                logs[f'gpu_id: {gpu_id}/{x} ({unit})'] = stats[i][j]
        return logs

    def _get_gpu_stats(self, queries: List[str]) -> List[List[float]]:
        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ','.join(queries)
        format = 'csv,nounits,noheader'
        result = subprocess.run(
            [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'--id={self._gpu_ids}'],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.

        stats = result.stdout.strip().split(os.linesep)
        stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
        return stats

    @staticmethod
    def _parse_gpu_stats(gpu_ids: str, stats: List[List[float]], keys: List[Tuple[str, str]]) -> Dict[str, float]:
        """Parse the gpu stats into a loggable dict"""
        logs = {}
        for i, gpu_id in enumerate(gpu_ids.split(',')):
            for j, (x, unit) in enumerate(keys):
                logs[f'gpu_id: {gpu_id}/{x} ({unit})'] = stats[i][j]
        return logs


def get_rgb_depth_datasets():
    """
    Method to create the datasets for the RGB autoencoder
    :return: Dict consisting of train-, val and test-set
    """
    train_dataset = RGBDepthDataset(
        mode='train',
        dataset_dir=config.dataset_dir
    )
    val_dataset = RGBDepthDataset(
        mode='val',
        dataset_dir=config.dataset_dir
    )
    test_dataset = RGBDepthDataset(
        mode='test',
        dataset_dir=config.dataset_dir
    )
    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def get_cifar10_datasets():
    """
    Method to create CIFAR10 datasets for the VGG16
    :return: Dict consisting of train-, val- and test-set
    """
    # download
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=False, download=True, transform=transform_test)

    # train/val split
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(trainset, [int(0.8 * len(trainset)), int(0.2 * len(trainset))])
    torch.manual_seed(torch.initial_seed())

    # undo data augmentation on val set
    val_dataset.transforms = transform_test

    assert len(train_dataset) + len(val_dataset) == len(trainset)

    return {'train': train_dataset, 'val': val_dataset, 'test': testset}


def checkParams(model):
    """
    Print number of model parameters
    """
    n_params = sum(p.numel() for p in model.parameters())
    print("FYI: Your model has {:.3f} params.".format(n_params / 1e6))


def save_reconstruction(reconstruction, save_dir, alpha, idx):
    """
    Save an (from an autoencoder reconstructed) image
    :param reconstruction: Image to save
    :param save_dir: Directory to save image
    :param alpha: Alpha-value with which the model (autoencoder) was trained
    :param idx: Index of image to save
    """
    # Revert preprocessing steps
    reconstruction = reconstruction.permute(1, 2, 0).numpy().astype(np.uint8)
    reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_RGB2BGR)

    reconstruction = cv2.resize(reconstruction, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(save_dir, str(alpha) + '_output_' + str(idx) + '.png'), reconstruction)
