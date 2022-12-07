#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

from ml_collections import config_dict


def get_configs(args, **kwargs):
    """Create config dicts for dataset, model and optimizer."""
    data_config = config_dict.ConfigDict()
    data_config.root = args.data_path
    data_config.name = args.dataset.lower()
    # minimum number of instances per class
    data_config.min_samples = args.min_samples
    # dataset imbalance is a function of p
    data_config.class_probs = kwargs.pop("p_mass")
    # number of classes that occur frequently in the data
    data_config.n_frequent_classes = 3
    # whether to balance mini-batches
    data_config.sampling = args.sampling
    # maximum number of triplets
    data_config.max_triplets = kwargs.pop("max_triplets")
    # input dimensionality
    data_config.input_dim = kwargs.pop("input_dim")
    # average number of instances per class
    M = kwargs.pop("n_samples")
    data_config.n_samples = M
    data_config.n_classes = args.n_classes
    data_config.k = args.k
    data_config.targets = args.targets

    data_config.oko_batch_size = kwargs.pop("oko_batch_size")
    data_config.main_batch_size = kwargs.pop("main_batch_size")

    model_config = config_dict.ConfigDict()
    model_config.type = re.compile(r"[a-zA-Z]+").search(args.network).group()

    try:
        model_config.depth = re.compile(r"\d+").search(args.network).group()
    except AttributeError:
        model_config.depth = ""

    model_config.regularization = args.regularization
    if args.regularization:
        if args.network.lower().startswith("resnet"):
            model_config.weight_decay = 1e-3
        else:
            model_config.weight_decay = 1e-4
    model_config.n_classes = args.n_classes

    if data_config.k == 0:
        model_config.task == "Pair"
    else:
        model_config.task = f"Odd-$k$-out ($k$={data_config.k}; {data_config.targets})"

    # TODO: enable half precision when running things on TPU
    model_config.half_precision = False

    optimizer_config = config_dict.ConfigDict()
    optimizer_config.name = args.optim
    optimizer_config.burnin = args.burnin
    optimizer_config.patience = args.patience
    optimizer_config.lr = kwargs.pop("eta")
    optimizer_config.epochs = kwargs.pop("epochs")
    data_config.initial_lr = optimizer_config.lr

    if optimizer_config.name.lower() == "sgd":
        # add momentum param if optim is sgd
        optimizer_config.momentum = 0.9

    # make config dicts immutable (same type as model param dicts)
    freeze = lambda cfg: config_dict.FrozenConfigDict(cfg)
    # freeze = lambda cfg: flax.core.frozen_dict.FrozenDict(cfg)
    data_config = freeze(data_config)
    model_config = freeze(model_config)
    optimizer_config = freeze(optimizer_config)
    return data_config, model_config, optimizer_config
