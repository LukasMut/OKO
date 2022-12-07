#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from einops import rearrange

import train
import utils
from config import get_configs

os.environ['PYTHONIOENCODING'] = "UTF-8"
os.environ['JAX_PLATFORM_NAME'] = "gpu"

tf.config.experimental.set_visible_devices([], 'GPU')

def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--out_path', type=str,
        help='path/to/params')
    aa('--data_path', type=str,
        help='path/to/original/dataset')
    aa('--dataset', type=str,
        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagenet'])
    aa('--network', type=str, default='ResNet18',
        choices=utils.MODELS)
    aa('--samples', type=int, nargs='+',
        help='average number of samples per class')
    aa('--n_classes', type=int,
        help='number of classes in dataset')
    aa('--k', type=int, default=1,
        choices=list(range(10)),
        help='number of odd classes in a set of k+2 examples with 2 examples coming from the same class')
    aa('--targets', type=str, default="hard",
        choices=["hard", "soft"],
        help="whether to use hard targets with a point mass at the pair class or soft targets that reflect the true class distribution in a set")
    aa('--oko_batch_sizes', type=int, nargs='+',
        help='number of sets per mini-batch (i.e., number of subsamples x 3')
    aa('--main_batch_sizes', type=int, nargs='+',
        help='number of triplets per mini-batch (i.e., number of subsamples x 3')
    aa('--max_triplets', type=int, nargs='+',
        help='maximum number of triplets during each epoch')
    aa('--probability_masses', type=float, nargs='+',
        help='probability mass that will be equally distributed among the k most frequent classes')
    aa('--epochs', type=int, nargs='+',
        help='maximum number of epochs')
    aa('--etas', type=float, nargs='+',
        help='learning rate for optimizer')
    aa('--optim', type=str, default='sgd',
        choices=['adam', 'adamw', 'radam', 'sgd', 'rmsprop'])
    aa('--burnin', type=int, default=30,
        help='burnin period before which convergence criterion is not evaluated (is equal to min number of epochs')
    aa('--patience', type=int, default=15,
        help='Number of steps of no improvement before stopping training')
    aa('--steps', type=int,
        help='save intermediate parameters every <steps> epochs')
    aa('--sampling', type=str, default='uniform',
        choices=['uniform', 'dynamic'],
        help='how to sample mini-batches per iteration')
    aa('--min_samples', type=int, default=None,
        help='minimum number of samples per class')
    aa('--testing', type=str, default='uniform',
        choices=['uniform', 'heterogeneous'],
        help='whether class prior probability at test time should be uniform or similar to training')
    aa('--regularization', action='store_true',
        help='apply l2 regularization during training')
    aa('--inference', action='store_true',
        help='whether to perform inference without stepping over the data')
    aa('--collect_reps', action='store_true',
        help='whether to store encoder latent representations')
    aa('--seeds', type=int, nargs='+',
        help='list of random seeds for cross-validating results over different random inits')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # get current combination of settings
    (n_samples, epochs, oko_batch_size, main_batch_size, eta, max_triplets), p_mass, rnd_seed = train.get_combination(
        samples=args.samples,
        epochs=args.epochs,
        oko_batch_sizes=args.oko_batch_sizes,
        main_batch_sizes=args.main_batch_sizes,
        learning_rates=args.etas,
        max_triplets=args.max_triplets,
        probability_masses=args.probability_masses,
        seeds=args.seeds,
        )

    # seed rng
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    # get few-shot subsets
    train_set, val_set = utils.get_fewshot_subsets(
        args,
        n_samples=n_samples,
        probability_mass=p_mass,
        rnd_seed=rnd_seed,
        )

    input_dim = train_set[0].shape[1:]
    num_classes = train_set[1].shape[-1]

    data_config, model_config, optimizer_config = get_configs(
        args,
        n_samples=n_samples,
        input_dim=input_dim,
        epochs=epochs, 
        oko_batch_size=oko_batch_size,
        main_batch_size=main_batch_size,
        max_triplets=max_triplets,
        p_mass=p_mass,
        eta=eta,
        )

    model = train.get_model(
        model_config=model_config,
        data_config=data_config
        )

    dir_config = train.create_dirs(
         results_root=args.out_path,
         data_config=data_config,
         model_config=model_config,
         rnd_seed=rnd_seed,
     )

    trainer, metrics, epoch = train.run(
        model=model,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        dir_config=dir_config,
        train_set=train_set,
        val_set=val_set,
        epochs=epochs,
        steps=args.steps,
        rnd_seed=rnd_seed,
        inference=args.inference,
        )

    if args.dataset == 'cifar10':
        dataset = np.load(
            os.path.join(args.data_path, 'test.npz')
        )
        images = dataset['data']
        labels = dataset['labels']
    else:
        dataset = torch.load(
            os.path.join(args.data_path, 'test.pt')
        )
        images = dataset[0].numpy()
        labels = dataset[1].numpy()

    if args.dataset.endswith('mnist'):
        X_test = rearrange(
            images, 'n h (w c) -> n h w c', c=1,
        )
    else:
        X_test = images

    y_test = jax.nn.one_hot(x=labels, num_classes=jnp.max(labels) + 1)

    train.inference(
        out_path=args.out_path,
        epoch=epoch,
        trainer=trainer,
        X_test=X_test,
        y_test=y_test,
        train_labels=train_set[1],
        model_config=model_config,
        data_config=data_config,
        dir_config=dir_config, 
        batch_size=main_batch_size,
        collect_reps=args.collect_reps,
    )
