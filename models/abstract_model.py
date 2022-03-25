# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SampleDown2(nn.Module):
    """
    Module consisting of two Conv2DBatchNorm followed by a maxpool.
    Used in the encoder in the RGB Autoencoder.
    """
    def __init__(self, map_location, in_channels, out_channels):
        super(SampleDown2, self).__init__()
        self.map_location = map_location
        self.model = nn.Sequential(
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        out = self.model(x)
        return out


class SampleDown3(nn.Module):
    """
    Module consisting of three Conv2DBatchNorm followed by a maxpool.
    Used in the encoder in the RGB Autoencoder.
    """
    def __init__(self, map_location, in_channels, out_channels):
        super(SampleDown3, self).__init__()
        self.map_location = map_location
        self.model = nn.Sequential(
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        out = self.model(x)
        return out


class SampleUp2(nn.Module):
    """
    Module consisting of a transposed convolution followed by two Conv2DBatchNorm.
    Used in the decoder in the RGB Autoencoder.
    """
    def __init__(self, map_location, in_channels, out_channels):
        super(SampleUp2, self).__init__()
        self.map_location = map_location
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=2,
                               stride=2),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        out = self.model(x)
        return out


class SampleUp3(nn.Module):
    """
    Module consisting of a transposed convolution followed by three Conv2DBatchNorm.
    Used in the decoder in the RGB Autoencoder.
    """
    def __init__(self, map_location, in_channels, out_channels):
        super(SampleUp3, self).__init__()
        self.map_location = map_location
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2,
                               stride=2),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            Conv2DBatchNorm(map_location=map_location,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        out = self.model(x)
        return out


class Conv2DBatchNorm(nn.Module):
    """
    Module consisting of a nn.Conv2D followed by a Batch Normalization.
    Used for VGG16 and the RGB Autoencoder.
    """
    def __init__(self, map_location, in_channels, out_channels, kernel_size=3, stride=1, padding=1, modules=None):
        super(Conv2DBatchNorm, self).__init__()
        self.map_location = map_location
        if modules is None:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.model = nn.Sequential(
                modules[0],
                modules[1],
            )

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        out = self.model(x)
        return out


class Flatten(nn.Module):
    def __init__(self, map_location):
        super(Flatten, self).__init__()
        self.map_location = map_location

    def forward(self, x):
        x = x.to(torch.device(self.map_location))
        x_shape = x.shape[0]
        return x.view(x_shape, -1)


class AbstractModel(pl.LightningModule):
    """
    Abstract superclass for all models
    """
    def __init__(self, hparams):
        super(AbstractModel, self).__init__()
        self.hparams = hparams

        self.activation = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(2, 2)
        self.avg_pooling = nn.AvgPool2d(1, 1)

    @abstractmethod
    def forward(self, x):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override forward()!')

    @abstractmethod
    def compute_loss(self, out, targets):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override compute_loss()!')

    def general_step(self, batch, batch_idx, mode):
        """
        Execution of forward-pass and loss computation for training, validation and testing.
        To be implemented in subclasses.
        :param batch: Current batch
        :param batch_idx: Current batch id
        :param mode: Current mode, i.e. train, val or test
        :return: loss and - if available - accuracy
        """
        raise NotImplementedError('subclasses must override general_step()!')

    def training_step(self, batch, batch_idx):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override training_step()!')

    def validation_step(self, batch, batch_idx):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override validation_step()!')

    def validation_epoch_end(self, outputs):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override validation_epoch_end()!')

    def test_step(self, batch, batch_idx):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override test_step()!')

    def test_epoch_end(self, outputs):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override test_epoch_end()!')

    @abstractmethod
    def prepare_data(self):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override prepare_data()!')

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], shuffle=True, batch_size=self.hparams['train_batch_size'],
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.hparams['val_batch_size'], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], num_workers=4)

    def configure_optimizers(self):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override configure_optimizers()!')

    def test(self, loader):
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override test()!')
