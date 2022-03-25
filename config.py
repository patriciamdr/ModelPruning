import os
import torch

"""
Configuration file for training, pruning and testing
"""

save_folder = 'models'
dataset_dir = '/storage/group/intellisys/datasets/carla/episode_000/'
default_rgb_data_dir = os.path.join(os.path.dirname(__file__), "rgb_eval")
default_depth_data_dir = os.path.join(os.path.dirname(__file__), "depth_eval")
default_rgb_save_dir = os.path.join(os.path.dirname(__file__), "rgb_eval_results")
default_depth_save_dir = os.path.join(os.path.dirname(__file__), "depth_eval_results")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.path.dirname(__file__), "data")
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
ckpts_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
patience = 25
max_epochs = 400
train_batch_size = 8
val_batch_size = 8
lr = 0.001
alpha = 1

remove_ratio = (1 - 0.65625)

# List of model params
vgg_params = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
autoencoder_params = [64, 128, 128, 128, 64]


def compute_pruned_vgg_params(remove_ratio):
    return [int(channel * remove_ratio) for channel in vgg_params]


def compute_pruned_autoencoder_params(remove_ratio, encoder, decoder):
    params = [int(channel * remove_ratio) for channel in autoencoder_params]
    if encoder and not decoder:
        return params[:3] + autoencoder_params[3:]
    elif not encoder and decoder:
        return autoencoder_params[:3] + params[3:]
    else:
        return params
