"""
File to prune parallel branches after reaching alpha=0 and obtaining the final pruned model.
"""
import argparse
import os
from copy import deepcopy

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import config
from models.autoencoder import Autoencoder
from models.autoencoder_dataset import RGBDepthDataset
from models.model_enum import Model
from models.pruned_model import PrunedModel
from models.vgg import VGGNet


def prune_weights(args, smaller_model, pruned_model):
    """
    Prune, i.e. multiply the parallel branches
    :param args: Command line arguments passed to this file, including class of pretrained model
    :param smaller_model: Final pruned model whose state_dict is updated with the new weights
    :param pruned_model: Model with parallel branches trained at alpha=0
    :return: Smaller model state dict with new smaller weights
    """
    pm_state_dict = pruned_model.state_dict()
    smaller_state_dict = deepcopy(smaller_model.state_dict())

    if isinstance(pruned_model.pretrained_model, VGGNet):
        # Iterate over all Conv2DBatchnorm layer except of last one
        for i in range(0, 13):
            if i == 0:
                # First Conv2DBatchnorm layer is special cases
                # First has no prev parallel branch
                # Compute weight of conv layer to prune
                print(pm_state_dict['pretrained_model.conv0.model.0.weight'].shape)
                print(pm_state_dict['parallel_branches.0.model.0.weight'].shape)
                print(smaller_state_dict['conv0.model.0.weight'].shape)

                pretrained_weights = pm_state_dict['pretrained_model.conv0.model.0.weight']
                pruned_weights = pm_state_dict['parallel_branches.0.model.0.weight']

                smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)

                smaller_state_dict['conv0.model.0.weight'] = smaller_weights

                # Compute corresponding bias
                print(pm_state_dict['pretrained_model.conv0.model.0.bias'].shape)
                print(pm_state_dict['parallel_branches.0.model.0.bias'].shape)
                print(smaller_state_dict['conv0.model.0.bias'].shape)

                pretrained_bias = pm_state_dict['pretrained_model.conv0.model.0.bias']
                pruned_bias = pm_state_dict['parallel_branches.0.model.0.bias']
                smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias

                smaller_state_dict['conv0.model.0.bias'] = smaller_bias
            else:
                # For remaining layers prev parallel branch has to be considered
                # Compute weight of conv layer to prune
                print(pm_state_dict['parallel_branches.' + str(i * 2 - 1) + '.weight'].shape)
                print(pm_state_dict['pretrained_model.conv' + str(i) + '.model.0.weight'].shape)
                print(pm_state_dict['parallel_branches.' + str(i * 2) + '.model.0.weight'].shape)
                print(smaller_state_dict['conv' + str(i) + '.model.0.weight'].shape)

                prev_pruned_weights = pm_state_dict['parallel_branches.' + str(i * 2 - 1) + '.weight']
                pretrained_weights = pm_state_dict['pretrained_model.conv' + str(i) + '.model.0.weight']
                pruned_weights = pm_state_dict['parallel_branches.' + str(i * 2) + '.model.0.weight']

                smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
                smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

                smaller_state_dict['conv' + str(i) + '.model.0.weight'] = smaller_weights

                # Compute corresponding bias
                print(pm_state_dict['parallel_branches.' + str(i * 2 - 1) + '.bias'].shape)
                print(pm_state_dict['pretrained_model.conv' + str(i) + '.model.0.bias'].shape)
                print(pm_state_dict['parallel_branches.' + str(i * 2) + '.model.0.bias'].shape)
                print(smaller_state_dict['conv' + str(i) + '.model.0.bias'].shape)

                prev_pruned_bias = pm_state_dict['parallel_branches.' + str(i * 2 - 1) + '.bias']
                pretrained_bias = pm_state_dict['pretrained_model.conv' + str(i) + '.model.0.bias']
                pruned_bias = pm_state_dict['parallel_branches.' + str(i * 2) + '.model.0.bias']
                smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights, prev_pruned_bias) \
                               + torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias

                smaller_state_dict['conv' + str(i) + '.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['conv' + str(i) + '.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(i * 2) + '.model.1.weight']
            smaller_state_dict['conv' + str(i) + '.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(i * 2) + '.model.1.bias']
            smaller_state_dict['conv' + str(i) + '.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(i * 2) + '.model.1.running_mean']
            smaller_state_dict['conv' + str(i) + '.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(i * 2) + '.model.1.running_var']
            smaller_state_dict['conv' + str(i) + '.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(i * 2) + '.model.1.num_batches_tracked']

        # Adjust linear layer weights
        print(pm_state_dict['parallel_branches.25.weight'].shape)
        print(pm_state_dict['pretrained_model.dense.weight'].shape)
        print(smaller_state_dict['dense.weight'].shape)

        pruned_weights = pm_state_dict['parallel_branches.25.weight']
        pretrained_weights = pm_state_dict['pretrained_model.dense.weight']

        smaller_weights = torch.matmul(pretrained_weights, pruned_weights)

        smaller_state_dict['dense.weight'] = smaller_weights

        # Compute corresponding bias
        print(pm_state_dict['parallel_branches.25.bias'].shape)
        print(pm_state_dict['pretrained_model.dense.bias'].shape)
        print(smaller_state_dict['dense.bias'].shape)

        pruned_bias = pm_state_dict['parallel_branches.25.bias']
        pretrained_bias = pm_state_dict['pretrained_model.dense.bias']
        smaller_bias = torch.matmul(pretrained_weights, pruned_bias) + pretrained_bias

        smaller_state_dict['dense.bias'] = smaller_bias

    elif isinstance(pruned_model.pretrained_model, Autoencoder):
        # Current parallel branch index
        if args.prune_encoder:
            curr_id = 0
            print("Prune encoder")
            # Iterate over all SampleDown layer

            # First sample down
            # First Conv2dBatchNorm
            # Compute weights
            print(pm_state_dict['pretrained_model.sample_down.0.model.0.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.0.model.0.weight'].shape)
            pretrained_weights = pm_state_dict['pretrained_model.sample_down.0.model.0.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.0.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            print(smaller_weights.shape)

            # Compute bias
            print(pm_state_dict['pretrained_model.sample_down.0.model.0.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.0.model.0.bias'].shape)
            pretrained_bias = pm_state_dict['pretrained_model.sample_down.0.model.0.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.0.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.0.model.0.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.0.model.0.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.0.model.0.model.1.weight'] = \
                pm_state_dict['parallel_branches.0.model.1.weight']
            smaller_state_dict['sample_down.0.model.0.model.1.bias'] = \
                pm_state_dict['parallel_branches.0.model.1.bias']
            smaller_state_dict['sample_down.0.model.0.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.0.model.1.running_mean']
            smaller_state_dict['sample_down.0.model.0.model.1.running_var'] = \
                pm_state_dict['parallel_branches.0.model.1.running_var']
            smaller_state_dict['sample_down.0.model.0.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.0.model.1.num_batches_tracked']

            # Second Conv2dBatchNorm
            # Compute weights
            print(pm_state_dict['parallel_branches.1.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.0.model.2.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.2.model.0.weight'].shape)
            snd_pruned_weights = pm_state_dict['parallel_branches.1.weight']
            snd_pretrained_weights = pm_state_dict['pretrained_model.sample_down.0.model.2.model.0.weight']
            third_pruned_weights = pm_state_dict['parallel_branches.2.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", third_pruned_weights, snd_pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, snd_pruned_weights)
            print(smaller_weights.shape)

            # Compute bias
            print(pm_state_dict['parallel_branches.1.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.0.model.2.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.2.model.0.bias'].shape)
            snd_pruned_bias = pm_state_dict['parallel_branches.1.bias']
            snd_pretrained_bias = pm_state_dict['pretrained_model.sample_down.0.model.2.model.0.bias']
            third_pruned_bias = pm_state_dict['parallel_branches.2.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', third_pruned_weights, snd_pretrained_bias) + third_pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', third_pruned_weights, snd_pretrained_weights, snd_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.0.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.0.model.2.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.0.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.2.model.1.weight']
            smaller_state_dict['sample_down.0.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.2.model.1.bias']
            smaller_state_dict['sample_down.0.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.2.model.1.running_mean']
            smaller_state_dict['sample_down.0.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.2.model.1.running_var']
            smaller_state_dict['sample_down.0.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.2.model.1.num_batches_tracked']

            # second sample down
            # first conv2dbatchnorm
            # compute weights
            print(pm_state_dict['parallel_branches.3.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.1.model.0.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.4.model.0.weight'].shape)
            prev_pruned_weights = pm_state_dict['parallel_branches.3.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_down.1.model.0.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.4.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)
            print(smaller_weights.shape)

            # compute bias
            print(pm_state_dict['parallel_branches.3.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.1.model.0.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.4.model.0.bias'].shape)
            prev_pruned_bias = pm_state_dict['parallel_branches.3.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_down.1.model.0.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.4.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights, prev_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.1.model.0.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.1.model.0.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.1.model.0.model.1.weight'] = \
                pm_state_dict['parallel_branches.4.model.1.weight']
            smaller_state_dict['sample_down.1.model.0.model.1.bias'] = \
                pm_state_dict['parallel_branches.4.model.1.bias']
            smaller_state_dict['sample_down.1.model.0.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.4.model.1.running_mean']
            smaller_state_dict['sample_down.1.model.0.model.1.running_var'] = \
                pm_state_dict['parallel_branches.4.model.1.running_var']
            smaller_state_dict['sample_down.1.model.0.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.4.model.1.num_batches_tracked']

            # second conv2dbatchnorm
            # compute weights
            print(pm_state_dict['parallel_branches.5.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.1.model.2.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.6.model.0.weight'].shape)
            snd_pruned_weights = pm_state_dict['parallel_branches.5.weight']
            snd_pretrained_weights = pm_state_dict['pretrained_model.sample_down.1.model.2.model.0.weight']
            third_pruned_weights = pm_state_dict['parallel_branches.6.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", third_pruned_weights, snd_pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, snd_pruned_weights)
            print(smaller_weights.shape)

            # compute bias
            print(pm_state_dict['parallel_branches.5.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.1.model.2.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.6.model.0.bias'].shape)
            snd_pruned_bias = pm_state_dict['parallel_branches.5.bias']
            snd_pretrained_bias = pm_state_dict['pretrained_model.sample_down.1.model.2.model.0.bias']
            third_pruned_bias = pm_state_dict['parallel_branches.6.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', third_pruned_weights, snd_pretrained_bias) + third_pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', third_pruned_weights, snd_pretrained_weights, snd_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.1.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.1.model.2.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.1.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.6.model.1.weight']
            smaller_state_dict['sample_down.1.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.6.model.1.bias']
            smaller_state_dict['sample_down.1.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.6.model.1.running_mean']
            smaller_state_dict['sample_down.1.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.6.model.1.running_var']
            smaller_state_dict['sample_down.1.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.6.model.1.num_batches_tracked']

            # Third sample down
            # First Conv2dBatchNorm
            # Compute weights
            print(pm_state_dict['parallel_branches.7.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.0.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.8.model.0.weight'].shape)
            prev_pruned_weights = pm_state_dict['parallel_branches.7.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_down.2.model.0.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.8.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)
            print(smaller_weights.shape)

            # Compute bias
            print(pm_state_dict['parallel_branches.7.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.0.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.8.model.0.bias'].shape)
            prev_pruned_bias = pm_state_dict['parallel_branches.7.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_down.2.model.0.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.8.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights, prev_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.2.model.0.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.2.model.0.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.2.model.0.model.1.weight'] = \
                pm_state_dict['parallel_branches.8.model.1.weight']
            smaller_state_dict['sample_down.2.model.0.model.1.bias'] = \
                pm_state_dict['parallel_branches.8.model.1.bias']
            smaller_state_dict['sample_down.2.model.0.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.8.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.0.model.1.running_var'] = \
                pm_state_dict['parallel_branches.8.model.1.running_var']
            smaller_state_dict['sample_down.2.model.0.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.8.model.1.num_batches_tracked']

            # Second Conv2dBatchNorm
            # Compute weights
            print(pm_state_dict['parallel_branches.9.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.2.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.10.model.0.weight'].shape)
            snd_pruned_weights = pm_state_dict['parallel_branches.9.weight']
            snd_pretrained_weights = pm_state_dict['pretrained_model.sample_down.2.model.2.model.0.weight']
            third_pruned_weights = pm_state_dict['parallel_branches.10.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", third_pruned_weights, snd_pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, snd_pruned_weights)
            print(smaller_weights.shape)

            # Compute bias
            print(pm_state_dict['parallel_branches.9.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.2.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.10.model.0.bias'].shape)
            snd_pruned_bias = pm_state_dict['parallel_branches.9.bias']
            snd_pretrained_bias = pm_state_dict['pretrained_model.sample_down.2.model.2.model.0.bias']
            third_pruned_bias = pm_state_dict['parallel_branches.10.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', third_pruned_weights, snd_pretrained_bias) + third_pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', third_pruned_weights, snd_pretrained_weights, snd_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.2.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.2.model.2.model.0.bias'] = smaller_bias

            smaller_state_dict['sample_down.2.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.10.model.1.weight']
            smaller_state_dict['sample_down.2.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.10.model.1.bias']
            smaller_state_dict['sample_down.2.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.10.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.10.model.1.running_var']
            smaller_state_dict['sample_down.2.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.10.model.1.num_batches_tracked']

            # Consider third pretrained convolutional layer in third SampleDown
            # Compute weights
            print(pm_state_dict['parallel_branches.11.weight'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.4.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.12.model.0.weight'].shape)
            fourth_pruned_weights = pm_state_dict['parallel_branches.11.weight']
            third_pretrained_weights = pm_state_dict['pretrained_model.sample_down.2.model.4.model.0.weight']
            fifth_pruned_weights = pm_state_dict['parallel_branches.12.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", fifth_pruned_weights, third_pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, fourth_pruned_weights)
            print(smaller_weights.shape)

            # Compute bias
            print(pm_state_dict['parallel_branches.11.bias'].shape)
            print(pm_state_dict['pretrained_model.sample_down.2.model.4.model.0.bias'].shape)
            print(pm_state_dict['parallel_branches.12.model.0.bias'].shape)
            fourth_pruned_bias = pm_state_dict['parallel_branches.11.bias']
            third_pretrained_bias = pm_state_dict['pretrained_model.sample_down.2.model.4.model.0.bias']
            fifth_pruned_bias = pm_state_dict['parallel_branches.12.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', fifth_pruned_weights, third_pretrained_bias) + fifth_pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', fifth_pruned_weights, third_pretrained_weights, fourth_pruned_bias) + smaller_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_down.2.model.4.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_down.2.model.4.model.0.bias'] = smaller_bias

            # Update BN
            smaller_state_dict['sample_down.2.model.4.model.1.weight'] = \
                pm_state_dict['parallel_branches.12.model.1.weight']
            smaller_state_dict['sample_down.2.model.4.model.1.bias'] = \
                pm_state_dict['parallel_branches.12.model.1.bias']
            smaller_state_dict['sample_down.2.model.4.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.12.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.4.model.1.running_var'] = \
                pm_state_dict['parallel_branches.12.model.1.running_var']
            smaller_state_dict['sample_down.2.model.4.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.12.model.1.num_batches_tracked']

            # Handle transposed convolution
            print(pm_state_dict['pretrained_model.sample_up.0.model.0.weight'].shape)
            print(pm_state_dict['parallel_branches.13.weight'].shape)
            fourth_pretrained_weight = pm_state_dict['pretrained_model.sample_up.0.model.0.weight']
            sixth_pruned_weight = pm_state_dict['parallel_branches.13.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", sixth_pruned_weight, fourth_pretrained_weight)
            print(smaller_weights.shape)

            print(pm_state_dict['pretrained_model.sample_up.0.model.0.bias'].shape)
            fourth_pretrained_bias = pm_state_dict['pretrained_model.sample_up.0.model.0.bias']
            # No bias in transposed conv of parallel branch
            smaller_bias = fourth_pretrained_bias
            print(smaller_bias.shape)

            smaller_state_dict['sample_up.0.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.0.model.0.bias'] = smaller_bias
            curr_id = 13
        else:
            curr_id = None
            print("Copy encoder")
            # Copy encoder weights
            smaller_state_dict['sample_down.0.model.0.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.0.weight']
            smaller_state_dict['sample_down.0.model.0.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.0.bias']
            smaller_state_dict['sample_down.0.model.0.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.1.weight']
            smaller_state_dict['sample_down.0.model.0.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.1.bias']
            smaller_state_dict['sample_down.0.model.0.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.1.running_mean']
            smaller_state_dict['sample_down.0.model.0.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.1.running_var']
            smaller_state_dict['sample_down.0.model.0.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.0.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.0.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.0.weight']
            smaller_state_dict['sample_down.0.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.0.bias']
            smaller_state_dict['sample_down.0.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.1.weight']
            smaller_state_dict['sample_down.0.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.1.bias']
            smaller_state_dict['sample_down.0.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.1.running_mean']
            smaller_state_dict['sample_down.0.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.1.running_var']
            smaller_state_dict['sample_down.0.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.0.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.1.model.0.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.0.weight']
            smaller_state_dict['sample_down.1.model.0.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.0.bias']
            smaller_state_dict['sample_down.1.model.0.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.1.weight']
            smaller_state_dict['sample_down.1.model.0.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.1.bias']
            smaller_state_dict['sample_down.1.model.0.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.1.running_mean']
            smaller_state_dict['sample_down.1.model.0.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.1.running_var']
            smaller_state_dict['sample_down.1.model.0.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.0.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.1.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.0.weight']
            smaller_state_dict['sample_down.1.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.0.bias']
            smaller_state_dict['sample_down.1.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.1.weight']
            smaller_state_dict['sample_down.1.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.1.bias']
            smaller_state_dict['sample_down.1.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.1.running_mean']
            smaller_state_dict['sample_down.1.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.1.running_var']
            smaller_state_dict['sample_down.1.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.1.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.2.model.0.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.0.weight']
            smaller_state_dict['sample_down.2.model.0.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.0.bias']
            smaller_state_dict['sample_down.2.model.0.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.1.weight']
            smaller_state_dict['sample_down.2.model.0.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.1.bias']
            smaller_state_dict['sample_down.2.model.0.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.0.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.1.running_var']
            smaller_state_dict['sample_down.2.model.0.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.0.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.2.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.0.weight']
            smaller_state_dict['sample_down.2.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.0.bias']
            smaller_state_dict['sample_down.2.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.1.weight']
            smaller_state_dict['sample_down.2.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.1.bias']
            smaller_state_dict['sample_down.2.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.1.running_var']
            smaller_state_dict['sample_down.2.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_down.2.model.4.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.0.weight']
            smaller_state_dict['sample_down.2.model.4.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.0.bias']
            smaller_state_dict['sample_down.2.model.4.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.1.weight']
            smaller_state_dict['sample_down.2.model.4.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.1.bias']
            smaller_state_dict['sample_down.2.model.4.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.1.running_mean']
            smaller_state_dict['sample_down.2.model.4.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.1.running_var']
            smaller_state_dict['sample_down.2.model.4.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_down.2.model.4.model.1.num_batches_tracked']
        if args.prune_decoder:
            print("Prune decoder")
            if curr_id is None:
                curr_id = -1
            # Iterate over all SampleUp layer except

            # First SampleUp
            # First Conv2dBatchNorm
            # Start with transposed conv
            # Compute weights
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.0.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 1) + '.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aecd", pretrained_weights, pruned_weights)

            # Compute bias
            # No bias in transposed conv of parallel branch
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.0.model.0.bias']
            smaller_bias = torch.einsum('abcd, a -> b', pruned_weights, pretrained_bias)

            # If encoder is pruned, than previous parallel branches need to be considered
            if args.prune_encoder:
                prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id) + '.weight']
                smaller_weights = torch.einsum("abcd, befg ->  aefg", prev_pruned_weights, smaller_weights)

                # No bias in transposed conv of parallel branch
                # smaller_bias = torch.einsum('abcd, befg -> e', pretrained_weights, pruned_weights) + smaller_bias

            smaller_state_dict['sample_up.0.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.0.model.0.bias'] = smaller_bias

            # First Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 2) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.0.model.2.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 2) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.0.model.2.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.0.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.0.model.2.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.0.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.1.weight']
            smaller_state_dict['sample_up.0.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.1.bias']
            smaller_state_dict['sample_up.0.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.1.running_var']
            smaller_state_dict['sample_up.0.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 3) + '.model.1.num_batches_tracked']

            # Second Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 4) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.0.model.4.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 4) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.0.model.4.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.0.model.4.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.0.model.4.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.0.model.4.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.1.weight']
            smaller_state_dict['sample_up.0.model.4.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.1.bias']
            smaller_state_dict['sample_up.0.model.4.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.4.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.1.running_var']
            smaller_state_dict['sample_up.0.model.4.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 5) + '.model.1.num_batches_tracked']

            # Third Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 6) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.0.model.6.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 6) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.0.model.6.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.0.model.6.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.0.model.6.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.0.model.6.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.1.weight']
            smaller_state_dict['sample_up.0.model.6.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.1.bias']
            smaller_state_dict['sample_up.0.model.6.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.6.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.1.running_var']
            smaller_state_dict['sample_up.0.model.6.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 7) + '.model.1.num_batches_tracked']

            # Second SampleUp
            # First Conv2dBatchNorm
            # Start with transposed conv
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 8) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.1.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 9) + '.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aecd", pretrained_weights, pruned_weights)
            try:
                smaller_weights = torch.einsum("abcd, befg ->  aefg", prev_pruned_weights, smaller_weights)
            except RuntimeError:
                smaller_weights = torch.einsum("abcd, befg ->  aefg", prev_pruned_weights.transpose(0, 1), smaller_weights)

            # Compute bias
            # No bias in transposed conv of parallel branch
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.1.model.0.bias']
            smaller_bias = torch.einsum('abcd, a -> b', pruned_weights, pretrained_bias)
            # smaller_bias = torch.einsum('abcd, befg -> a', pruned_weights, pretrained_weights) + smaller_bias
            # smaller_bias = torch.einsum('abcd, befg -> e', pretrained_weights, pruned_weights) + smaller_bias

            smaller_state_dict['sample_up.1.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.1.model.0.bias'] = smaller_bias

            # First Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 10) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.1.model.2.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 10) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.1.model.2.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.1.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.1.model.2.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.1.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.1.weight']
            smaller_state_dict['sample_up.1.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.1.bias']
            smaller_state_dict['sample_up.1.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.1.running_mean']
            smaller_state_dict['sample_up.1.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.1.running_var']
            smaller_state_dict['sample_up.1.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 11) + '.model.1.num_batches_tracked']

            # Second Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 12) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.1.model.4.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 12) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.1.model.4.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.1.model.4.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.1.model.4.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.1.model.4.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.1.weight']
            smaller_state_dict['sample_up.1.model.4.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.1.bias']
            smaller_state_dict['sample_up.1.model.4.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.1.running_mean']
            smaller_state_dict['sample_up.1.model.4.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.1.running_var']
            smaller_state_dict['sample_up.1.model.4.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 13) + '.model.1.num_batches_tracked']

            # Third SampleUp
            # First Conv2dBatchNorm
            # Start with transposed conv
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 14) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.2.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 15) + '.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aecd", pretrained_weights, pruned_weights)
            try:
                smaller_weights = torch.einsum("abcd, befg ->  aefg", prev_pruned_weights, smaller_weights)
            except RuntimeError:
                smaller_weights = torch.einsum("abcd, befg ->  aefg", prev_pruned_weights.transpose(0, 1), smaller_weights)

            # Compute bias
            # No bias in transposed conv of parallel branch
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.2.model.0.bias']
            smaller_bias = torch.einsum('abcd, a -> b', pruned_weights, pretrained_bias)
            # smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights) + smaller_bias
            # smaller_bias = torch.einsum('abcd, befg -> e', pretrained_weights, pruned_weights) + smaller_bias

            smaller_state_dict['sample_up.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.2.model.0.bias'] = smaller_bias

            # First Conv2dBatchNorm
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 16) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.2.model.2.model.0.weight']
            pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.0.weight']
            smaller_weights = torch.einsum("abcd, befg ->  aefg", pruned_weights, pretrained_weights)
            smaller_weights = torch.einsum("abcd, befg ->  aecd", smaller_weights, prev_pruned_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 16) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.2.model.2.model.0.bias']
            pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pruned_weights, pretrained_bias) + pruned_bias
            smaller_bias = torch.einsum('abcd, befg, e -> a', pruned_weights, pretrained_weights,
                                        prev_pruned_bias) + smaller_bias

            smaller_state_dict['sample_up.2.model.2.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.2.model.2.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.2.model.2.model.1.weight'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.1.weight']
            smaller_state_dict['sample_up.2.model.2.model.1.bias'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.1.bias']
            smaller_state_dict['sample_up.2.model.2.model.1.running_mean'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.1.running_mean']
            smaller_state_dict['sample_up.2.model.2.model.1.running_var'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.1.running_var']
            smaller_state_dict['sample_up.2.model.2.model.1.num_batches_tracked'] = \
                pm_state_dict['parallel_branches.' + str(curr_id + 17) + '.model.1.num_batches_tracked']

            # Second Conv2dBatchNorm
            # No following outgoing parallel branch
            # Compute weights
            prev_pruned_weights = pm_state_dict['parallel_branches.' + str(curr_id + 18) + '.weight']
            pretrained_weights = pm_state_dict['pretrained_model.sample_up.2.model.4.model.0.weight']
            smaller_weights = torch.einsum("abcd, eafg ->  ebfg", prev_pruned_weights, pretrained_weights)

            # Compute bias
            prev_pruned_bias = pm_state_dict['parallel_branches.' + str(curr_id + 18) + '.bias']
            pretrained_bias = pm_state_dict['pretrained_model.sample_up.2.model.4.model.0.bias']
            smaller_bias = torch.einsum('abcd, b -> a', pretrained_weights, prev_pruned_bias) + pretrained_bias

            smaller_state_dict['sample_up.2.model.4.model.0.weight'] = smaller_weights
            smaller_state_dict['sample_up.2.model.4.model.0.bias'] = smaller_bias

            # Update batchnorm
            smaller_state_dict['sample_up.2.model.4.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.weight']
            smaller_state_dict['sample_up.2.model.4.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.bias']
            smaller_state_dict['sample_up.2.model.4.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.running_mean']
            smaller_state_dict['sample_up.2.model.4.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.running_var']
            smaller_state_dict['sample_up.2.model.4.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.num_batches_tracked']

        else:
            print("Copy decoder")
            # Copy decoder weights
            smaller_state_dict['sample_up.0.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.0.weight']
            smaller_state_dict['sample_up.0.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.0.bias']
            smaller_state_dict['sample_up.0.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.1.weight']
            smaller_state_dict['sample_up.0.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.1.bias']
            smaller_state_dict['sample_up.0.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.1.running_var']
            smaller_state_dict['sample_up.0.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.0.model.4.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.0.weight']
            smaller_state_dict['sample_up.0.model.4.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.0.bias']
            smaller_state_dict['sample_up.0.model.4.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.1.weight']
            smaller_state_dict['sample_up.0.model.4.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.1.bias']
            smaller_state_dict['sample_up.0.model.4.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.4.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.1.running_var']
            smaller_state_dict['sample_up.0.model.4.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.4.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.0.model.6.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.0.weight']
            smaller_state_dict['sample_up.0.model.6.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.0.bias']
            smaller_state_dict['sample_up.0.model.6.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.1.weight']
            smaller_state_dict['sample_up.0.model.6.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.1.bias']
            smaller_state_dict['sample_up.0.model.6.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.1.running_mean']
            smaller_state_dict['sample_up.0.model.6.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.1.running_var']
            smaller_state_dict['sample_up.0.model.6.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.0.model.6.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.1.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.0.weight']
            smaller_state_dict['sample_up.1.model.0.bias'] = pm_state_dict['pretrained_model.sample_up.1.model.0.bias']
            smaller_state_dict['sample_up.1.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.0.weight']
            smaller_state_dict['sample_up.1.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.0.bias']
            smaller_state_dict['sample_up.1.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.1.weight']
            smaller_state_dict['sample_up.1.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.1.bias']
            smaller_state_dict['sample_up.1.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.1.running_mean']
            smaller_state_dict['sample_up.1.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.1.running_var']
            smaller_state_dict['sample_up.1.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.1.model.4.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.0.weight']
            smaller_state_dict['sample_up.1.model.4.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.0.bias']
            smaller_state_dict['sample_up.1.model.4.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.1.weight']
            smaller_state_dict['sample_up.1.model.4.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.1.bias']
            smaller_state_dict['sample_up.1.model.4.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.1.running_mean']
            smaller_state_dict['sample_up.1.model.4.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.1.running_var']
            smaller_state_dict['sample_up.1.model.4.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.1.model.4.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.0.weight']
            smaller_state_dict['sample_up.2.model.0.bias'] = pm_state_dict['pretrained_model.sample_up.2.model.0.bias']
            smaller_state_dict['sample_up.2.model.2.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.0.weight']
            smaller_state_dict['sample_up.2.model.2.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.0.bias']
            smaller_state_dict['sample_up.2.model.2.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.1.weight']
            smaller_state_dict['sample_up.2.model.2.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.1.bias']
            smaller_state_dict['sample_up.2.model.2.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.1.running_mean']
            smaller_state_dict['sample_up.2.model.2.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.1.running_var']
            smaller_state_dict['sample_up.2.model.2.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.2.model.1.num_batches_tracked']

            smaller_state_dict['sample_up.2.model.4.model.0.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.0.weight']
            smaller_state_dict['sample_up.2.model.4.model.0.bias'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.0.bias']
            smaller_state_dict['sample_up.2.model.4.model.1.weight'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.weight']
            smaller_state_dict['sample_up.2.model.4.model.1.bias'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.bias']
            smaller_state_dict['sample_up.2.model.4.model.1.running_mean'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.running_mean']
            smaller_state_dict['sample_up.2.model.4.model.1.running_var'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.running_var']
            smaller_state_dict['sample_up.2.model.4.model.1.num_batches_tracked'] = pm_state_dict[
                'pretrained_model.sample_up.2.model.4.model.1.num_batches_tracked']
    return smaller_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruning_path',
                        default=os.path.join(config.ckpts_dir, 'VGG/Step/pruned-train-original/alpha=0.0_epoch=74-val_loss=0.38.ckpt'),
                        help='Path to model to prune')
    parser.add_argument('--model',
                        help='Name of pretrained model',
                        default='VGG')
    parser.add_argument('--pretrained_path',
                        default=os.path.join(config.ckpts_dir, 'VGG/original_epoch=149-val_loss=0.34.ckpt'),
                        help='Path to pretrained model')
    parser.add_argument('--encoder_pruned', dest='encoder_pruned', action='store_true',
                        help='Whether the encoder of the pretrained model is already pruned  or not')
    parser.set_defaults(encoder_pruned=False)
    parser.add_argument('--train_original', dest='train_original', action='store_true',
                        help='Whether to train the pretrained model or not')
    parser.set_defaults(train_original=True)
    parser.add_argument('--prune_encoder', dest='prune_encoder', action='store_true',
                        help='Whether to prune the encoder or not')
    parser.set_defaults(prune_encoder=False)
    parser.add_argument('--prune_decoder', dest='prune_decoder', action='store_true',
                        help='Whether to prune the decoder or not')
    parser.set_defaults(prune_decoder=False)
    args = parser.parse_args()
    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'lr': config.lr,
        'train_original': args.train_original,
        'dataset_dir': config.data_dir,
        'map_location': "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    print(args.pretrained_path)
    checkpoint = torch.load(args.pretrained_path, map_location=torch.device(hparams['map_location']))

    # Load original and final smaller model (last ones weight still have to be updated)
    if args.model is Model.VGG.value:
        pretrained_model = VGGNet(hparams=hparams)
        pretrained_model = pretrained_model.to(torch.device(hparams['map_location']))
        smaller_model = VGGNet(hparams=hparams, model_params=config.compute_pruned_vgg_params(config.remove_ratio))
        smaller_model = smaller_model.to(torch.device(hparams['map_location']))
        state_dict = checkpoint['state_dict']
    elif args.model is Model.Autoencoder.value:
        hparams['type'] = 'rgb'
        hparams['prune_encoder'] = args.prune_encoder
        hparams['prune_decoder'] = args.prune_decoder
        if args.encoder_pruned:
            state_dict = checkpoint
            # Pretrained model's encoder is already pruned
            params = config.compute_pruned_autoencoder_params(config.remove_ratio, encoder=True, decoder=False)
            smaller_params = config.compute_pruned_autoencoder_params(config.remove_ratio, encoder=True, decoder=hparams['prune_decoder'])
        else:
            state_dict = checkpoint['state_dict']
            # Pretrained model is unpruned
            params = config.autoencoder_params
            smaller_params = config.compute_pruned_autoencoder_params(config.remove_ratio, encoder=hparams['prune_encoder'], decoder=hparams['prune_decoder'])

        pretrained_model = Autoencoder(hparams=hparams, model_params=params)
        pretrained_model = pretrained_model.to(torch.device(hparams['map_location']))
        smaller_model = Autoencoder(hparams=hparams, model_params=smaller_params)
        smaller_model = smaller_model.to(torch.device(hparams['map_location']))

    pretrained_model.load_state_dict(state_dict)

    # Load model which parallel branches should be pruned
    ckpt = torch.load(args.pruning_path, map_location=torch.device(hparams['map_location']))
    state_dict = ckpt['state_dict']
    hparams['alpha'] = 0
    pruned_model = PrunedModel(hparams=hparams, pretrained_model=pretrained_model)
    pruned_model = pruned_model.to(torch.device(hparams['map_location']))
    pruned_model.load_state_dict(state_dict)

    # Prune parallel branches and update smaller model
    smaller_state_dict = prune_weights(args, smaller_model, pruned_model)
    smaller_model.load_state_dict(smaller_state_dict)

    # Test smaller model to check whether pruning has been done correctly
    if args.model is Model.VGG.value:
        trainer = pl.Trainer(gpus=0)
        trainer.test(smaller_model)
    elif args.model is Model.Autoencoder.value:
        test_dataset = RGBDepthDataset(mode='test')
        test_dataloader = DataLoader(test_dataset)

        print("Test smaller")
        smaller_model.eval()
        smaller_model.freeze()
        smaller_model.output_reconstruction(test_dataloader, -1)

    # Save new smaller model
    dir_name = '/'.join(args.pruning_path.split('/')[:-1])
    torch.save(smaller_model.state_dict(), os.path.join(dir_name, 'smaller_model.ckpt'))
