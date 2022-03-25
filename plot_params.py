"""
File to collect and plot the accuracy loss at different alpha-values wrt to the original model accuracy when pruning the VGG
"""
import glob
import re
import config
import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from models.pruned_model import PrunedModel
from models.vgg import VGGNet

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root=config.data_dir, train=False, download=True, transform=transform_test)


def collect_acc(hparams):
    """
    Test the original VGG as well as each intermediate model for every alpha-iteration.
    Save the accuracy of each csv-file.
    :param hparams: Hyperparameters for the models
    """
    checkpoint = os.path.join(config.ckpts_dir, 'VGG/original_epoch=149-val_loss=0.34.ckpt')
    pretrained_model = VGGNet(hparams=hparams)
    test_dataloader = DataLoader(testset, num_workers=2)
    pretrained_ckpt = torch.load(checkpoint, map_location=config.device)
    pretrained_state_dict = pretrained_ckpt['state_dict']
    pretrained_model.load_state_dict(pretrained_state_dict)
    pretrained_model = pretrained_model.to(torch.device(hparams['map_location']))

    # Test original model
    new_data = pd.DataFrame()
    original_test_acc = pretrained_model.test(test_dataloader)
    df = pd.DataFrame([{'alpha': 1.0,
                        'acc': original_test_acc}])
    print('Accuracy: ' + str(original_test_acc))
    new_data = new_data.append(df, ignore_index=True)

    # Iterate over every intermediate model
    list_of_files = glob.glob(os.path.join(config.ckpts_dir, 'VGG/Step/pruned-train-original/*.ckpt'))
    for i, path in enumerate(list_of_files):
        print(path)
        alpha = re.search('=(.+?)_', path)
        if alpha:
            alpha = alpha.group(1)
        else:
            continue
        print("Alpha: " + alpha)
        hparams['alpha'] = float(alpha)

        test_model = PrunedModel(hparams=hparams, pretrained_model=pretrained_model)
        checkpoint = torch.load(path, map_location=config.device)
        test_model.load_state_dict(checkpoint['state_dict'])
        test_model = test_model.to(torch.device(hparams['map_location']))
        test_model.eval()
        test_model.freeze()

        avg_test_acc = test_model.test(test_dataloader)

        df = pd.DataFrame([{'alpha': float(alpha),
                            'acc': avg_test_acc}])
        print('Accuracy: ' + str(avg_test_acc))
        new_data = new_data.append(df, ignore_index=True)

    # Save results
    if not os.path.exists(config.data_dir):
        os.mkdir(config.data_dir)
    new_data.to_csv(os.path.join(config.data_dir, 'test_acc.csv'), index=False, header=False)


def plot_params_vs_acc():
    """
    Plot accuracy loss vs alpha-value
    """
    data = pd.read_csv(os.path.join(config.data_dir, 'test_acc.csv'), names=['alpha', 'acc'])
    sb.set_style('whitegrid', {'grid.linestyle': '--'})
    sb.set_style('ticks')
    sb.set_context('paper')

    # Get original model accuracy
    original_acc = (data.loc[data['alpha'] == 1.0])['acc'].item()
    # Compute percentage of accuracy loss
    acc_loss = ((data['acc'] / original_acc) - 1) * 100
    data['acc_loss'] = acc_loss
    g = sb.lineplot(x='alpha', y='acc_loss', marker='o', data=data, ci=95)
    png_path = os.path.join(config.data_dir, 'params_pruned_vs_acc_loss.png')
    g.set(yticks=[1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0])
    g.set(xticks=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    plt.xlabel('α')
    plt.ylabel('Accuracy loss [%]')
    plt.savefig(png_path, format='png')
    plt.show()


if __name__ == '__main__':
    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'lr': config.lr,
        'train_original': True,
        'map_location': "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    collect_acc(hparams)
    plot_params_vs_acc()
