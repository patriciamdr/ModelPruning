"""
Model based on: https://github.com/foamliu/Autoencoder
Use transposed convolutions instead of unpooling.

Code based on I2DL exercises.
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ssim_loss import pytorch_ssim

import torch
import torch.nn as nn

import utils
import config
from models.abstract_model import AbstractModel, SampleDown2, SampleDown3, SampleUp3, SampleUp2


class Autoencoder(AbstractModel):
    def __init__(self, hparams, model_params=config.autoencoder_params):
        super(Autoencoder, self).__init__(hparams)

        self.sample_down = nn.Sequential(
            SampleDown2(map_location=hparams['map_location'],
                        in_channels=self.hparams['in_channels'],
                        out_channels=model_params[0]),
            SampleDown2(map_location=hparams['map_location'],
                        in_channels=model_params[0],
                        out_channels=model_params[1]),
            SampleDown3(map_location=hparams['map_location'],
                        in_channels=model_params[1],
                        out_channels=model_params[2])
        )
        self.sample_up = nn.Sequential(
            SampleUp3(map_location=hparams['map_location'],
                      in_channels=model_params[2],
                      out_channels=model_params[3]),
            SampleUp2(map_location=hparams['map_location'],
                      in_channels=model_params[3],
                      out_channels=model_params[4]),
            SampleUp2(map_location=hparams['map_location'],
                      in_channels=model_params[4],
                      out_channels=self.hparams['in_channels'])
        )

        self.l1_loss_func = nn.L1Loss()
        self.ssim_loss_func = pytorch_ssim.SSIM()

    def forward(self, x):
        x = x.to(torch.device(self.hparams['map_location']))
        out_down = self.sample_down(x)
        out_up = self.sample_up(out_down)
        return out_up

    @staticmethod
    def combine_losses(l1_loss, ssim_loss):
        return 0.5 * l1_loss + 0.5 * ssim_loss

    def compute_loss(self, out, targets):
        l1_loss = self.l1_loss_func(out, targets)
        # Use ssim loss as described in here: https://github.com/Po-Hsun-Su/pytorch-ssim
        ssim_out = -self.ssim_loss_func(out, targets)
        # Save ssim loss for logging
        ssim_loss = -ssim_out

        total_loss = self.combine_losses(l1_loss, ssim_out)

        return total_loss, l1_loss, ssim_loss

    def general_step(self, batch, batch_idx, mode):
        rgb, depth = batch['rgb'].to(self.hparams['map_location']), batch['depth'].to(self.hparams['map_location'])

        # Select on which images to train on
        if self.hparams['type'] == 'rgb':
            images = rgb
        elif self.hparams['type'] == 'depth':
            images = depth

        # In order to let input type and weight type be the same
        images = images.float()
        outputs = self.forward(images)

        total_loss, l1_loss, ssim_loss = self.compute_loss(outputs, images)

        return total_loss, l1_loss, ssim_loss

    @staticmethod
    def general_end(outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, 'train')
        tensorboard_logs = {'loss': loss, 'l1_loss': l1_loss, 'ssim_loss': ssim_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.general_step(batch, batch_idx, 'val')
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, 'val')
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        self.dataset = utils.get_rgb_depth_datasets()

    def configure_optimizers(self):
        params = list(self.parameters())
        optim = torch.optim.Adam(params=params, lr=self.hparams['lr'])
        scheduler1 = ReduceLROnPlateau(optim)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler1,
            'monitor': 'val_loss'
        }

    def test(self, loader):
        total_loss = 0
        for idx, batch in enumerate(loader):
            rgb, depth = batch['rgb'].to(self.hparams['map_location']), batch['depth'].to(self.hparams['map_location'])

            # Select on which images to train on
            if self.hparams['type'] == 'rgb':
                images = rgb
            elif self.hparams['type'] == 'depth':
                images = depth

            # In order to let input type and weight type be the same
            images = images.float()
            outputs = self.forward(images)
            loss, l1_loss, ssim_loss = self.compute_loss(outputs, images)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss

    def output_reconstruction(self, loader, i):
        for idx, batch in enumerate(loader):
            rgb, depth = batch['rgb'].to(self.hparams['map_location']), batch['depth'].to(self.hparams['map_location'])

            # Select on which images to train on
            if self.hparams['type'] == 'rgb':
                images = rgb
            elif self.hparams['type'] == 'depth':
                images = depth

            # In order to let input type and weight type be the same
            images = images.float()
            outputs = self.forward(images)
            total_loss, l1_loss, ssim_loss = self.compute_loss(outputs, images)
            print("total loss: " + str(total_loss))
            print("l1 loss: " + str(l1_loss))
            print("ssim loss: " + str(ssim_loss))

            utils.save_reconstruction(outputs[0], config.default_rgb_save_dir, i, idx)
