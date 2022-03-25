"""
File to execute the pruning process.
"""
import glob
import os

import argparse as argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from models.autoencoder import Autoencoder
from models.model_enum import Model
from models.pruned_model import PrunedModel
from models.regression_model import RegressionModel
from models.vgg import VGGNet
from scheduler.exponential_scheduler import ExponentialScheduler
from scheduler.multiplicative_scheduler import MultiplicativeScheduler
from scheduler.scheduler_enum import Scheduler
from scheduler.step_scheduler import StepScheduler


def load_pretrained_model(args):
    """
    Load the pretrained model
    :param args: Command line arguments passed to this file, including class of pretrained model
    :return: A pretrained model object
    """
    pretrained_ckpt = torch.load(args.pretrained_model_path, map_location=config.device)
    if args.encoder_pruned:
        pretrained_state_dict = pretrained_ckpt
    else:
        pretrained_state_dict = pretrained_ckpt['state_dict']

    # Create pretrained model object
    # For 'framework' concept, later more model selection
    if args.model is Model.VGG.value:
        pretrained_model = VGGNet(hparams=hparams)
    elif args.model is Model.Autoencoder.value:
        if args.encoder_pruned:
            # Pretrained model's encoder is already pruned
            params = config.compute_pruned_autoencoder_params(config.remove_ratio, encoder=True, decoder=False)
        else:
            # Pretrained model is unpruned
            params = config.autoencoder_params

        pretrained_model = Autoencoder(hparams=hparams, model_params=params)

    pretrained_model.load_state_dict(pretrained_state_dict)
    return pretrained_model


def load_pruned_model(args, hparams, pretrained_model, remove_ratio):
    """
    Load the complete pruned model, consisting of original model and parallel branches
    :param args: Command line arguments passed to this file, including class of pretrained model
    :param hparams: Hyperparameters
    :param pretrained_model: Pretrained model object
    :param remove_ratio: Ratio of how many parameters should be pruned away per layer
    :return: A pruned model object
    """
    if args.as_regression_task:
        return RegressionModel(hparams=hparams, pretrained_model=pretrained_model, remove_ratio=remove_ratio)
    else:
        if isinstance(pretrained_model, Autoencoder):
            hparams['type'] = 'rgb'
            hparams['prune_encoder'] = args.prune_encoder
            hparams['prune_decoder'] = args.prune_decoder
        return PrunedModel(hparams=hparams, pretrained_model=pretrained_model, remove_ratio=remove_ratio)


def load_scheduler(args):
    """
    Load a scheduler to schedule the alpha-decay
    :param args: Command line arguments passed to this file, including class of desired scheduler
    :return: A scheduler object
    """
    if args.scheduler == Scheduler.Step.value:
        return StepScheduler(step_size=args.schedule_param)
    elif args.scheduler == Scheduler.Mult.value:
        return MultiplicativeScheduler(mu=args.schedule_param)
    elif args.scheduler == Scheduler.Exp.value:
        return ExponentialScheduler(decay_rate=args.schedule_param)


def objective_selection(args, hparams, pretrained_model, scheduler):
    """
    Run the alpha-schedule loop until alpha=0
    :param args: Command line arguments passed to this file
    :param hparams: Hyperparameters
    :param pretrained_model: Pretrained model object
    :param scheduler: Alpha-scheduler object
    :return:
    """
    while True:
        # Get next alpha
        alpha = scheduler.step()
        # If alpha smaller than threshold, start last iteration with alpha=0
        if alpha <= args.threshold:
            alpha = 0
        print("Alpha: " + str(alpha))
        hparams['alpha'] = alpha

        # Specify target-directory to save intermediate results
        if args.as_regression_task:
            tb_name = 'regression/alpha=' + str(alpha)
            ckpt_dir = os.path.join(config.ckpts_dir, args.model, args.scheduler, 'regression/pruned')
        else:
            if args.train_original:
                tb_name = 'alpha=' + str(alpha) + '-train-original'
                ckpt_dir = os.path.join(config.ckpts_dir, args.model, args.scheduler, 'pruned-train-original')
            else:
                tb_name = 'alpha=' + str(alpha)
                ckpt_dir = os.path.join(config.ckpts_dir, args.model, args.scheduler, 'pruned')

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(config.logs_dir, args.model, args.scheduler),
                                                 name=tb_name)
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'alpha=' + str(alpha) + '_{epoch:02d}-{val_loss:.2f}'),
            monitor='val_loss',
            mode='min'
        )

        model = load_pruned_model(args, hparams, pretrained_model, args.remove_ratio)

        try:
            # Load latest model from target-directory
            list_of_files = glob.glob(ckpt_dir + '/*.ckpt')
            path = max(list_of_files, key=os.path.getctime)
            print(path)

            ckpt = torch.load(path, map_location=config.device)
            state_dict = ckpt['state_dict']
            model.load_state_dict(state_dict)

            print("Continue pruning with parallel branches")
        except (FileNotFoundError, NameError, ValueError) as e:
            print("Start first pruning iteration with parallel branches")

        trainer = pl.Trainer(
            logger=tb_logger,
            gpus=1,
            max_epochs=config.max_epochs,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model)
        # Stop objective selection
        if alpha <= 0:
            print("Stop objective selection")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='Name of the model to prune',
                        default='Autoencoder')
    parser.add_argument('--pretrained_model_path',
                        help='Path to pretrained model checkpoint',
                        default=os.path.join(config.ckpts_dir, 'Autoencoder/original_epoch=437-val_loss=1.40.ckpt'))
    parser.add_argument('--encoder_pruned', dest='encoder_pruned', action='store_true',
                        help='Whether the encoder of the pretrained model is already pruned  or not')
    parser.set_defaults(encoder_pruned=False)
    parser.add_argument('--remove_ratio',
                        default=config.remove_ratio,
                        type=float,
                        help="Percentage to prune")
    parser.add_argument('--scheduler',
                        help='Name of scheduler to schedule alpha',
                        default='Step')
    parser.add_argument('--schedule_param',
                        help='Parameter to pass to scheduler, dependent on type of scheduler',
                        default=0.1)
    parser.add_argument('--threshold',
                        default=1.5e-2,
                        help='Once alpha drops below this threshold start last iteration')
    parser.add_argument('--train_original', dest='train_original', action='store_true',
                        help='Whether to train the pretrained model or not')
    parser.set_defaults(train_original=True)
    parser.add_argument('--prune_encoder', dest='prune_encoder', action='store_true',
                        help='Whether to prune the encoder or not')
    parser.set_defaults(prune_encoder=False)
    parser.add_argument('--prune_decoder', dest='prune_decoder', action='store_true',
                        help='Whether to prune the decoder or not')
    parser.set_defaults(prune_decoder=False)
    parser.add_argument('--regression', dest='as_regression_task', action='store_true',
                        help='Train pruned models on logits of pretrained model')
    parser.set_defaults(as_regression_task=False)
    args = parser.parse_args()

    hparams = {
        'in_channels': 3,
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'lr': config.lr,
        'train_original': args.train_original,
        'map_location': config.device
    }

    pretrained_model = load_pretrained_model(args)
    scheduler = load_scheduler(args)
    objective_selection(args, hparams, pretrained_model, scheduler)
