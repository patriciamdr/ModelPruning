import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
import utils
from models.abstract_model import AbstractModel, Flatten, Conv2DBatchNorm
from models.autoencoder import Autoencoder
from models.vgg import VGGNet
from ssim_loss import pytorch_ssim


class PrunedModel(AbstractModel):
    def __init__(self, hparams, pretrained_model, remove_ratio=config.remove_ratio):
        super(PrunedModel, self).__init__(hparams)

        # Add parallel branches for every Conv2DBatchNorm layer
        self.parallel_branches = nn.ModuleList()

        if isinstance(pretrained_model, VGGNet):
            self.loss = nn.CrossEntropyLoss()

            items = list(pretrained_model._modules.items())
            for idx, (name, module) in enumerate(items):
                if isinstance(module, Conv2DBatchNorm):
                    num_channels = module.model[0].out_channels
                    pruned_channels = int(module.model[0].out_channels * remove_ratio)
                    self.parallel_branches.append(Conv2DBatchNorm(map_location=self.hparams['map_location'],
                                                                  in_channels=num_channels,
                                                                  out_channels=pruned_channels,
                                                                  kernel_size=1,
                                                                  stride=1,
                                                                  padding=0))

                    # If following module is conv, last layer in parallel branch is also conv
                    if isinstance(items[idx + 1][1], Conv2DBatchNorm):
                        self.parallel_branches.append(nn.Conv2d(in_channels=pruned_channels,
                                                                out_channels=num_channels,
                                                                kernel_size=1,
                                                                stride=1,
                                                                padding=0))
                    # If following module is fc, last layer in parallel branch is also fc
                    elif isinstance(items[idx + 1][1], nn.Linear):
                        self.parallel_branches.append(nn.Linear(in_features=pruned_channels,
                                                                out_features=num_channels))

        elif isinstance(pretrained_model, Autoencoder):
            self.l1_loss_func = nn.L1Loss()
            self.ssim_loss_func = pytorch_ssim.SSIM()

            self.prune_encoder = hparams['prune_encoder']
            self.prune_decoder = hparams['prune_decoder']

            reached_bottleneck = False
            items = [module for module in pretrained_model.modules() if type(module) != nn.Sequential]
            # Last conv layer (incl. bn and relu) will not be pruned)
            for idx, module in enumerate(items):
                if isinstance(module, nn.Conv2d):
                    # Only prune decoder, do not add parallel branches for convs in encoder
                    if self.prune_decoder and not self.prune_encoder and not reached_bottleneck:
                        continue
                    num_channels = module.out_channels
                    pruned_channels = int(num_channels * remove_ratio)
                    self.parallel_branches.append(Conv2DBatchNorm(map_location=self.hparams['map_location'],
                                                                  in_channels=num_channels,
                                                                  out_channels=pruned_channels,
                                                                  kernel_size=1,
                                                                  stride=1,
                                                                  padding=0))
                    self.parallel_branches.append(nn.Conv2d(in_channels=pruned_channels,
                                                            out_channels=num_channels,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0))
                # If following layer is transposed conv, change last layer in parallel branch to transposed conv
                elif isinstance(module, nn.ConvTranspose2d):
                    reached_bottleneck = True
                    num_channels = module.out_channels
                    pruned_channels = int(num_channels * remove_ratio)
                    if self.prune_encoder:
                        self.parallel_branches[-1] = nn.ConvTranspose2d(in_channels=pruned_channels,
                                out_channels=num_channels,
                                kernel_size=1,
                                stride=1,
                                bias=False)
                        # Only prune encoder, stop adding parallel branches
                        if not self.prune_decoder:
                            break

                    # Add parallel branch
                    self.parallel_branches.append(nn.ConvTranspose2d(in_channels=num_channels,
                                                                     out_channels=pruned_channels,
                                                                     kernel_size=1,
                                                                     stride=1,
                                                                     bias=False))
                    # Following layers are standard convolutions
                    self.parallel_branches.append(nn.Conv2d(in_channels=pruned_channels,
                                                            out_channels=num_channels,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0))

        self.apply(utils.weights_init)

        self.pretrained_model = pretrained_model
        for params in self.pretrained_model.parameters():
            params.requires_grad = bool(self.hparams['train_original'])

        self.alpha = self.hparams['alpha']

    def prepare_data(self):
        # assign to use in dataloaders
        if isinstance(self.pretrained_model, VGGNet):
            self.dataset = utils.get_cifar10_datasets()
        elif isinstance(self.pretrained_model, Autoencoder):
            self.dataset = utils.get_rgb_depth_datasets()

    def compute_loss(self, out, targets):
        if isinstance(self.pretrained_model, VGGNet):
            return self.loss(out, targets)
        elif isinstance(self.pretrained_model, Autoencoder):
            l1_loss = self.l1_loss_func(out, targets)
            # Use ssim loss as described in here: https://github.com/Po-Hsun-Su/pytorch-ssim
            ssim_out = -self.ssim_loss_func(out, targets)
            # Save ssim loss for logging
            ssim_loss = -ssim_out

            total_loss = 0.5 * (l1_loss + ssim_loss)

            return total_loss, l1_loss, ssim_loss

    def general_step(self, batch, batch_idx, mode):
        if isinstance(self.pretrained_model, VGGNet):
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
        elif isinstance(self.pretrained_model, Autoencoder):
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

    def forward(self, x):
        x = x.to(torch.device(self.hparams['map_location']))
        if isinstance(self.pretrained_model, VGGNet):
            out0 = self.pretrained_model.conv0.model[0](x)
            # Parallel branch
            pout0 = self.parallel_branches[0](out0)
            pout0 = self.activation(pout0)
            pout0 = self.parallel_branches[1](pout0)

            out0 = self.pretrained_model.conv0.model[1](out0)
            out0 = self.activation(out0)
            out = self.alpha * out0 + (1 - self.alpha) * pout0

            out1 = self.pretrained_model.conv1.model[0](out)
            # Parallel branch
            pout1 = self.parallel_branches[2](out1)
            pout1 = self.activation(pout1)
            pout1 = self.max_pooling(pout1)
            pout1 = self.parallel_branches[3](pout1)

            out1 = self.pretrained_model.conv1.model[1](out1)
            out1 = self.activation(out1)
            out1 = self.max_pooling(out1)
            out = self.alpha * out1 + (1 - self.alpha) * pout1

            out2 = self.pretrained_model.conv2.model[0](out)
            # Parallel branch
            pout2 = self.parallel_branches[4](out2)
            pout2 = self.activation(pout2)
            pout2 = self.parallel_branches[5](pout2)

            out2 = self.pretrained_model.conv2.model[1](out2)
            out2 = self.activation(out2)
            out = self.alpha * out2 + (1 - self.alpha) * pout2

            out3 = self.pretrained_model.conv3.model[0](out)
            # Parallel branch
            pout3 = self.parallel_branches[6](out3)
            pout3 = self.activation(pout3)
            pout3 = self.max_pooling(pout3)
            pout3 = self.parallel_branches[7](pout3)

            out3 = self.pretrained_model.conv3.model[1](out3)
            out3 = self.activation(out3)
            out3 = self.max_pooling(out3)
            out = self.alpha * out3 + (1 - self.alpha) * pout3

            out4 = self.pretrained_model.conv4.model[0](out)
            # Parallel branch
            pout4 = self.parallel_branches[8](out4)
            pout4 = self.activation(pout4)
            pout4 = self.parallel_branches[9](pout4)

            out4 = self.pretrained_model.conv4.model[1](out4)
            out4 = self.activation(out4)
            out = self.alpha * out4 + (1 - self.alpha) * pout4

            out5 = self.pretrained_model.conv5.model[0](out)
            # Parallel branch
            pout5 = self.parallel_branches[10](out5)
            pout5 = self.activation(pout5)
            pout5 = self.parallel_branches[11](pout5)

            out5 = self.pretrained_model.conv5.model[1](out5)
            out5 = self.activation(out5)
            out = self.alpha * out5 + (1 - self.alpha) * pout5

            out6 = self.pretrained_model.conv6.model[0](out)
            # Parallel branch
            pout6 = self.parallel_branches[12](out6)
            pout6 = self.activation(pout6)
            pout6 = self.max_pooling(pout6)
            pout6 = self.parallel_branches[13](pout6)

            out6 = self.pretrained_model.conv6.model[1](out6)
            out6 = self.activation(out6)
            out6 = self.max_pooling(out6)
            out = self.alpha * out6 + (1 - self.alpha) * pout6

            out7 = self.pretrained_model.conv7.model[0](out)
            # Parallel branch
            pout7 = self.parallel_branches[14](out7)
            pout7 = self.activation(pout7)
            pout7 = self.parallel_branches[15](pout7)

            out7 = self.pretrained_model.conv7.model[1](out7)
            out7 = self.activation(out7)
            out = self.alpha * out7 + (1 - self.alpha) * pout7

            out8 = self.pretrained_model.conv8.model[0](out)
            # Parallel branch
            pout8 = self.parallel_branches[16](out8)
            pout8 = self.activation(pout8)
            pout8 = self.parallel_branches[17](pout8)

            out8 = self.pretrained_model.conv8.model[1](out8)
            out8 = self.activation(out8)
            out = self.alpha * out8 + (1 - self.alpha) * pout8

            out9 = self.pretrained_model.conv9.model[0](out)
            # Parallel branch
            pout9 = self.parallel_branches[18](out9)
            pout9 = self.activation(pout9)
            pout9 = self.max_pooling(pout9)
            pout9 = self.parallel_branches[19](pout9)

            out9 = self.pretrained_model.conv9.model[1](out9)
            out9 = self.activation(out9)
            out9 = self.max_pooling(out9)
            out = self.alpha * out9 + (1 - self.alpha) * pout9

            out10 = self.pretrained_model.conv10.model[0](out)
            # Parallel branch
            pout10 = self.parallel_branches[20](out10)
            pout10 = self.activation(pout10)
            pout10 = self.parallel_branches[21](pout10)

            out10 = self.pretrained_model.conv10.model[1](out10)
            out10 = self.activation(out10)
            out = self.alpha * out10 + (1 - self.alpha) * pout10

            out11 = self.pretrained_model.conv11.model[0](out)
            # Parallel branch
            pout11 = self.parallel_branches[22](out11)
            pout11 = self.activation(pout11)
            pout11 = self.parallel_branches[23](pout11)

            out11 = self.pretrained_model.conv11.model[1](out11)
            out11 = self.activation(out11)
            out = self.alpha * out11 + (1 - self.alpha) * pout11

            out12 = self.pretrained_model.conv12.model[0](out)
            # Parallel branch
            pout12 = self.parallel_branches[24](out12)
            pout12 = self.activation(pout12)
            pout12 = self.max_pooling(pout12)
            pout12 = self.avg_pooling(pout12)
            pout12 = Flatten(self.hparams['map_location'])(pout12)
            pout12 = self.parallel_branches[25](pout12)

            out12 = self.pretrained_model.conv12.model[1](out12)
            out12 = self.activation(out12)
            out12 = self.max_pooling(out12)
            out12 = self.avg_pooling(out12)
            out12 = Flatten(self.hparams['map_location'])(out12)

            out = self.alpha * out12 + (1 - self.alpha) * pout12

            out = self.pretrained_model.dense(out)
        elif isinstance(self.pretrained_model, Autoencoder):
            curr_id = 0
            if self.prune_encoder:
                out0 = self.pretrained_model.sample_down[0].model[0].model[0](x)
                # Parallel branch
                pout0 = self.parallel_branches[curr_id](out0)
                curr_id = curr_id + 1
                pout0 = self.activation(pout0)
                pout0 = self.parallel_branches[curr_id](pout0)
                curr_id = curr_id + 1

                out0 = self.pretrained_model.sample_down[0].model[0].model[1](out0)
                out0 = self.pretrained_model.sample_down[0].model[1](out0)
                out = self.alpha * out0 + (1 - self.alpha) * pout0

                out1 = self.pretrained_model.sample_down[0].model[2].model[0](out)
                # Parallel branch
                pout1 = self.parallel_branches[curr_id](out1)
                curr_id = curr_id + 1
                pout1 = self.activation(pout1)
                pout1 = self.max_pooling(pout1)
                pout1 = self.parallel_branches[curr_id](pout1)
                curr_id = curr_id + 1

                out1 = self.pretrained_model.sample_down[0].model[2].model[1](out1)
                out1 = self.pretrained_model.sample_down[0].model[3](out1)
                out1 = self.pretrained_model.sample_down[0].model[4](out1)
                out = self.alpha * out1 + (1 - self.alpha) * pout1

                out2 = self.pretrained_model.sample_down[1].model[0].model[0](out)
                # Parallel branch
                pout2 = self.parallel_branches[curr_id](out2)
                curr_id = curr_id + 1
                pout2 = self.activation(pout2)
                pout2 = self.parallel_branches[curr_id](pout2)
                curr_id = curr_id + 1

                out2 = self.pretrained_model.sample_down[1].model[0].model[1](out2)
                out2 = self.pretrained_model.sample_down[1].model[1](out2)
                out = self.alpha * out2 + (1 - self.alpha) * pout2

                out3 = self.pretrained_model.sample_down[1].model[2].model[0](out)
                # Parallel branch
                pout3 = self.parallel_branches[curr_id](out3)
                curr_id = curr_id + 1
                pout3 = self.activation(pout3)
                pout3 = self.max_pooling(pout3)
                pout3 = self.parallel_branches[curr_id](pout3)
                curr_id = curr_id + 1

                out3 = self.pretrained_model.sample_down[1].model[2].model[1](out3)
                out3 = self.pretrained_model.sample_down[1].model[3](out3)
                out3 = self.pretrained_model.sample_down[1].model[4](out3)
                out = self.alpha * out3 + (1 - self.alpha) * pout3

                out4 = self.pretrained_model.sample_down[2].model[0].model[0](out)
                # Parallel branch
                pout4 = self.parallel_branches[curr_id](out4)
                curr_id = curr_id + 1
                pout4 = self.activation(pout4)
                pout4 = self.parallel_branches[curr_id](pout4)
                curr_id = curr_id + 1

                out4 = self.pretrained_model.sample_down[2].model[0].model[1](out4)
                out4 = self.pretrained_model.sample_down[2].model[1](out4)
                out = self.alpha * out4 + (1 - self.alpha) * pout4

                out5 = self.pretrained_model.sample_down[2].model[2].model[0](out)
                # Parallel branch
                pout5 = self.parallel_branches[curr_id](out5)
                curr_id = curr_id + 1
                pout5 = self.activation(pout5)
                pout5 = self.parallel_branches[curr_id](pout5)
                curr_id = curr_id + 1

                out5 = self.pretrained_model.sample_down[2].model[2].model[1](out5)
                out5 = self.pretrained_model.sample_down[2].model[3](out5)
                out = self.alpha * out5 + (1 - self.alpha) * pout5

                out6 = self.pretrained_model.sample_down[2].model[4].model[0](out)
                # Parallel branch
                pout6 = self.parallel_branches[curr_id](out6)
                curr_id = curr_id + 1
                pout6 = self.activation(pout6)
                pout6 = self.max_pooling(pout6)
                # Bottleneck
                pout6 = self.parallel_branches[curr_id](pout6)
                curr_id = curr_id + 1

                out6 = self.pretrained_model.sample_down[2].model[4].model[1](out6)
                out6 = self.pretrained_model.sample_down[2].model[5](out6)
                out6 = self.pretrained_model.sample_down[2].model[6](out6)
                out = self.alpha * out6 + (1 - self.alpha) * pout6
            else:
                out = self.pretrained_model.sample_down(x)

            if self.prune_decoder:
                # Transposed conv
                out7 = self.pretrained_model.sample_up[0].model[0](out)
                # Parallel branch
                pout7 = self.parallel_branches[curr_id](out7)
                curr_id = curr_id + 1
                pout7 = self.activation(pout7)
                pout7 = self.parallel_branches[curr_id](pout7)
                curr_id = curr_id + 1

                out7 = self.pretrained_model.sample_up[0].model[1](out7)
                out = self.alpha * out7 + (1 - self.alpha) * pout7

                out8 = self.pretrained_model.sample_up[0].model[2].model[0](out)
                # Parallel branch
                pout8 = self.parallel_branches[curr_id](out8)
                curr_id = curr_id + 1
                pout8 = self.activation(pout8)
                pout8 = self.parallel_branches[curr_id](pout8)
                curr_id = curr_id + 1

                out8 = self.pretrained_model.sample_up[0].model[2].model[1](out8)
                out8 = self.pretrained_model.sample_up[0].model[3](out8)
                out = self.alpha * out8 + (1 - self.alpha) * pout8

                out9 = self.pretrained_model.sample_up[0].model[4].model[0](out)
                # Parallel branch
                pout9 = self.parallel_branches[curr_id](out9)
                curr_id = curr_id + 1
                pout9 = self.activation(pout9)
                pout9 = self.parallel_branches[curr_id](pout9)
                curr_id = curr_id + 1

                out9 = self.pretrained_model.sample_up[0].model[4].model[1](out9)
                out9 = self.pretrained_model.sample_up[0].model[5](out9)
                out = self.alpha * out9 + (1 - self.alpha) * pout9

                out10 = self.pretrained_model.sample_up[0].model[6].model[0](out)
                # Parallel branch
                pout10 = self.parallel_branches[curr_id](out10)
                curr_id = curr_id + 1
                pout10 = self.activation(pout10)
                pout10 = self.parallel_branches[curr_id](pout10)
                curr_id = curr_id + 1

                out10 = self.pretrained_model.sample_up[0].model[6].model[1](out10)
                out10 = self.pretrained_model.sample_up[0].model[7](out10)
                out = self.alpha * out10 + (1 - self.alpha) * pout10

                # Transposed conv
                out11 = self.pretrained_model.sample_up[1].model[0](out)
                # Parallel branch
                pout11 = self.parallel_branches[curr_id](out11)
                curr_id = curr_id + 1
                pout11 = self.activation(pout11)
                pout11 = self.parallel_branches[curr_id](pout11)
                curr_id = curr_id + 1

                out11 = self.pretrained_model.sample_up[1].model[1](out11)
                out = self.alpha * out11 + (1 - self.alpha) * pout11

                out12 = self.pretrained_model.sample_up[1].model[2].model[0](out)
                # Parallel branch
                pout12 = self.parallel_branches[curr_id](out12)
                curr_id = curr_id + 1
                pout12 = self.activation(pout12)
                pout12 = self.parallel_branches[curr_id](pout12)
                curr_id = curr_id + 1

                out12 = self.pretrained_model.sample_up[1].model[2].model[1](out12)
                out12 = self.pretrained_model.sample_up[1].model[3](out12)
                out = self.alpha * out12 + (1 - self.alpha) * pout12

                out13 = self.pretrained_model.sample_up[1].model[4].model[0](out)
                # Parallel branch
                pout13 = self.parallel_branches[curr_id](out13)
                curr_id = curr_id + 1
                pout13 = self.activation(pout13)
                pout13 = self.parallel_branches[curr_id](pout13)
                curr_id = curr_id + 1

                out13 = self.pretrained_model.sample_up[1].model[4].model[1](out13)
                out13 = self.pretrained_model.sample_up[1].model[5](out13)
                out = self.alpha * out13 + (1 - self.alpha) * pout13

                # Transposed conv
                out14 = self.pretrained_model.sample_up[2].model[0](out)
                # Parallel branch
                pout14 = self.parallel_branches[curr_id](out14)
                curr_id = curr_id + 1
                pout14 = self.activation(pout14)
                pout14 = self.parallel_branches[curr_id](pout14)
                curr_id = curr_id + 1

                out14 = self.pretrained_model.sample_up[2].model[1](out14)
                out = self.alpha * out14 + (1 - self.alpha) * pout14

                out15 = self.pretrained_model.sample_up[2].model[2].model[0](out)
                # Parallel branch
                pout15 = self.parallel_branches[curr_id](out15)
                curr_id = curr_id + 1
                pout15 = self.activation(pout15)
                pout15 = self.parallel_branches[curr_id](pout15)
                curr_id = curr_id + 1

                out15 = self.pretrained_model.sample_up[2].model[2].model[1](out15)
                out15 = self.pretrained_model.sample_up[2].model[3](out15)
                out = self.alpha * out15 + (1 - self.alpha) * pout15

                out16 = self.pretrained_model.sample_up[2].model[4].model[0](out)
                out16 = self.pretrained_model.sample_up[2].model[4].model[1](out16)
                out = self.pretrained_model.sample_up[2].model[5](out16)
            else:
                out = self.pretrained_model.sample_up(out)
        return out

    @staticmethod
    def general_end(outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        if isinstance(self.pretrained_model, VGGNet):
            loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, 'train')
            tensorboard_logs = {'loss': loss, 'l1_loss': l1_loss, 'ssim_loss': ssim_loss}
            return {'loss': loss, 'log': tensorboard_logs}
        elif isinstance(self.pretrained_model, Autoencoder):
            loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, 'train')
            tensorboard_logs = {'loss': loss, 'alpha': self.alpha, 'l1_loss': l1_loss, 'ssim_loss': ssim_loss}
            return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        if isinstance(self.pretrained_model, VGGNet):
            loss, _, _ = self.general_step(batch, batch_idx, 'val')
            return {'val_loss': loss}
        elif isinstance(self.pretrained_model, Autoencoder):
            loss, _, _ = self.general_step(batch, batch_idx, 'val')
            return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        if isinstance(self.pretrained_model, VGGNet):
            avg_loss = self.general_end(outputs, 'val')
            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}
        elif isinstance(self.pretrained_model, Autoencoder):
            avg_loss = self.general_end(outputs, 'val')
            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        if isinstance(self.pretrained_model, VGGNet):
            loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, "test")
            return {'test_loss': loss}
        elif isinstance(self.pretrained_model, Autoencoder):
            loss, l1_loss, ssim_loss = self.general_step(batch, batch_idx, "test")
            return {'test_loss': loss}

    def test_epoch_step(self, outputs):
        if isinstance(self.pretrained_model, VGGNet):
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
            tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
            return {'test_loss': avg_loss, 'log': tensorboard_logs}
        elif isinstance(self.pretrained_model, Autoencoder):
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            tensorboard_logs = {'test_loss': avg_loss}
            return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if isinstance(self.pretrained_model, VGGNet):
            params = list(self.parameters())
            optim = torch.optim.SGD(params=params, lr=self.hparams['lr'], momentum=0.9)
            scheduler1 = ReduceLROnPlateau(optimizer=optim)
            return {
                'optimizer': optim,
                'lr_scheduler': scheduler1,
                'monitor': 'val_loss'
            }
        elif isinstance(self.pretrained_model, Autoencoder):
            params = list(self.parameters())
            optim = torch.optim.Adam(params=params, lr=self.hparams['lr'])
            scheduler1 = ReduceLROnPlateau(optim)
            return {
                'optimizer': optim,
                'lr_scheduler': scheduler1,
                'monitor': 'val_loss'
            }

    def test(self, loader):
        if isinstance(self.pretrained_model, VGGNet):
            total_acc = 0
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

            avg_acc = total_acc / len(loader)
            return avg_acc
        elif isinstance(self.pretrained_model, Autoencoder):
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
