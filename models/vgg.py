# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
import utils
from models.abstract_model import AbstractModel, Flatten, Conv2DBatchNorm


class VGGNet(AbstractModel):
    def __init__(self, hparams, model_params=config.vgg_params):
        super(VGGNet, self).__init__(hparams)
        self.loss = nn.CrossEntropyLoss()
        self.conv0 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=self.hparams['in_channels'],
                                     out_channels=model_params[0])

        self.conv1 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[0],
                                     out_channels=model_params[1])

        self.conv2 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[1],
                                     out_channels=model_params[2])

        self.conv3 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[2],
                                     out_channels=model_params[3])

        self.conv4 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[3],
                                     out_channels=model_params[4])

        self.conv5 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[4],
                                     out_channels=model_params[5])

        self.conv6 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[5],
                                     out_channels=model_params[6])

        self.conv7 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[6],
                                     out_channels=model_params[7])

        self.conv8 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[7],
                                     out_channels=model_params[8])

        self.conv9 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                     in_channels=model_params[8],
                                     out_channels=model_params[9])

        self.conv10 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                      in_channels=model_params[9],
                                      out_channels=model_params[10])

        self.conv11 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                      in_channels=model_params[10],
                                      out_channels=model_params[11])

        self.conv12 = Conv2DBatchNorm(map_location=torch.device(self.hparams['map_location']),
                                      in_channels=model_params[11],
                                      out_channels=model_params[12])

        self.dense = nn.Linear(in_features=model_params[12], out_features=10)

    def compute_loss(self, out, targets):
        return self.loss(out, targets)

    def prepare_data(self):
        # assign to use in dataloaders
        self.dataset = utils.get_cifar10_datasets()

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        images, targets = images.to(torch.device(self.hparams['map_location'])), targets.to(
            torch.device(self.hparams['map_location']))

        # forward pass
        out = self.forward(images)

        # loss
        loss = self.compute_loss(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)  # convert output probabilities to predicted class
        acc = preds.eq(targets).sum().float() / targets.size(0)

        return loss, acc

    def forward(self, x):
        x = x.to(torch.device(self.hparams['map_location']))
        out0 = self.conv0(x)
        out0 = self.activation(out0)
        out1 = self.conv1(out0)
        out1 = self.activation(out1)
        out1 = self.max_pooling(out1)

        out2 = self.conv2(out1)
        out2 = self.activation(out2)
        out3 = self.conv3(out2)
        out3 = self.activation(out3)
        out3 = self.max_pooling(out3)

        out4 = self.conv4(out3)
        out4 = self.activation(out4)
        out5 = self.conv5(out4)
        out5 = self.activation(out5)
        out6 = self.conv6(out5)
        out6 = self.activation(out6)
        out6 = self.max_pooling(out6)

        out7 = self.conv7(out6)
        out7 = self.activation(out7)
        out8 = self.conv8(out7)
        out8 = self.activation(out8)
        out9 = self.conv9(out8)
        out9 = self.activation(out9)
        out9 = self.max_pooling(out9)

        out10 = self.conv10(out9)
        out10 = self.activation(out10)
        out11 = self.conv11(out10)
        out11 = self.activation(out11)
        out12 = self.conv12(out11)
        out12 = self.activation(out12)
        out12 = self.max_pooling(out12)

        out12 = self.avg_pooling(out12)

        out = Flatten(self.hparams['map_location'])(out12)
        out = self.dense(out)

        return out

    def configure_optimizers(self):
        params = list(self.parameters())
        optim = torch.optim.SGD(params=params, lr=self.hparams['lr'], momentum=0.9)
        scheduler1 = ReduceLROnPlateau(optimizer=optim)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler1,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, 'train')
        tensorboard_logs = {'loss': loss, 'acc': acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, 'val')
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def test(self, loader):
        total_acc = 0
        num = 0
        for batch in loader:
            images, targets = batch
            images, targets = images.to(config.device), targets.to(config.device)

            # forward pass
            out = self.forward(images)

            # loss
            loss = self.loss(out, targets)

            # accuracy
            _, preds = torch.max(out, 1)  # convert output probabilities to predicted class
            acc = preds.eq(targets).sum().float() / targets.size(0)

            total_acc += acc.item()
            num += 1

        avg_acc = total_acc / num
        return avg_acc
