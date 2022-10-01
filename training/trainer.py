#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Tuple

import flax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict, freeze
from flax.training import checkpoints
from flax.training.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import training.utils as utils
from training.train_state import TrainState

Array = jnp.ndarray
Model = Any


@dataclass
class OOOTrainer:
    model: Model
    model_config: FrozenDict
    data_config: FrozenDict
    optimizer_config: FrozenDict
    dir_config: FrozenDict
    steps: int
    rnd_seed: int

    def __post_init__(self):
        self.rng_seq = hk.PRNGSequence(self.rnd_seed)
        self.rng = jax.random.PRNGKey(self.rnd_seed)
        # freeze model config dictionary (i.e., make it immutable)
        self.model_config = FrozenDict(self.model_config)

        if self.model_config["task"] == "mtl":
            self.tasks = ["mle", "ooo"]
        else:
            self.tasks = [self.model_config["task"]]

        # inititalize model
        self.init_model()

        self.logger = SummaryWriter(log_dir=self.dir_config.log_dir)
        self.early_stop = EarlyStopping(
            min_delta=1e-4, patience=self.optimizer_config.patience
        )
        # create jitted train and eval functions
        self.create_functions()

        if self.data_config.distribution == "heterogeneous":
            setattr(self, "class_hitting", True)

        self.train_metrics = list()
        self.test_metrics = list()

    def init_model(self) -> None:
        """Initialise parameters (i.e., weights and biases) of neural network."""
        key_i, key_j = random.split(random.PRNGKey(self.rnd_seed))

        if self.data_config.name.endswith("mnist"):
            H, W = self.data_config.input_dim
            C = 1  # gray channel
        else:
            H, W, C = self.data_config.input_dim

        def get_init_batch(batch_size, task):
            if task == "ooo":
                batch = random.normal(key_i, shape=(batch_size * 3, H, W, C))
            else:
                batch = random.normal(key_i, shape=(batch_size, H, W, C))
            return batch

        if self.model_config["type"].lower() == "resnet":
            for task in self.tasks:
                batch = get_init_batch(self.data_config.batch_size, task)
                variables = self.model.init(key_j, batch, train=True, task=task)
                init_params, self.init_batch_stats = (
                    variables["params"],
                    variables["batch_stats"],
                )
                setattr(self, f"init_{task}_params", init_params)
        else:
            if self.model_config["type"].lower() == "vit":
                for task in self.tasks:
                    batch = get_init_batch(self.data_config.batch_size, task)
                    self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
                    init_params = self.model.init(
                        {"params": init_rng, "dropout": dropout_init_rng},
                        batch,
                        train=True,
                        task=task,
                    )["params"]
                    setattr(self, f"init_{task}_params", init_params)
            else:
                for task in self.tasks:
                    batch = get_init_batch(self.data_config.batch_size, task)

                    """
                    # NOTE: this part is only necessary if we implement the triplet odd-one-out clf head as a Transformer
                    if task == 'ooo':
                        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
                        init_params = self.model.init(
                                {"params": init_rng, "dropout": dropout_init_rng},
                                batch,
                                task=task,
                            )["params"]
                        setattr(self, f"init_{task}_params", init_params)
                    else:
                        variables = self.model.init(key_j, batch, task=task)
                        _, init_params = variables.pop("params")
                        setattr(self, f"init_{task}_params", init_params)
                        del variables
                    """

                    variables = self.model.init(key_j, batch, task=task)
                    _, init_params = variables.pop("params")
                    setattr(self, f"init_{task}_params", init_params)
                    del variables
            self.init_batch_stats = None
        self.state = None

    def get_optim(self, train_batches: Iterator):
        opt_class = getattr(optax, self.optimizer_config.name)
        opt_hypers = {}
        # opt_hypers["learning_rate"] = self.optimizer_config.lr
        if self.optimizer_config.name.lower() == "sgd":
            opt_hypers["momentum"] = self.optimizer_config.momentum
            opt_hypers["nesterov"] = True

        # we decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_config.lr,
            boundaries_and_scales={
                int(len(train_batches) * self.optimizer_config.epochs * 0.6): 0.1,
                int(len(train_batches) * self.optimizer_config.epochs * 0.85): 0.1,
            },
        )
        # clip gradients at maximum value
        transf = [optax.clip(1.0)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **opt_hypers),
        )
        return optimizer

    def init_optim(self, train_batches: Iterator) -> None:
        """Initialize optimizer and training state."""
        optimizer = self.get_optim(train_batches)
        # initialize training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_mle_params
            if self.state is None
            else self.state.mle_params,
            batch_stats=self.init_batch_stats
            if self.state is None
            else self.state.batch_stats,
            tx=optimizer,
            task=self.model_config["task"],
            ooo_params=self.init_ooo_params
            if self.state is None
            else self.state.ooo_params,
        )

    def create_functions(self) -> None:
        def init_loss_fn(model_config: FrozenDict, state: Any, task: str) -> Any:
            if task == "ooo":
                # create all six permutations of positions
                perms = jax.device_put(
                    jnp.array(list(itertools.permutations(range(3), 3)))
                )
                loss_fn = partial(
                    getattr(utils, f"{task}_loss_fn_{model_config['type'].lower()}"),
                    state,
                    perms,
                )
            else:
                loss_fn = partial(
                    getattr(utils, f"{task}_loss_fn_{model_config['type'].lower()}"),
                    state,
                )
            return loss_fn

        def get_loss_fn(state, model_config: FrozenDict, train=None) -> Callable:
            """Get task and model specific loss function."""
            if model_config["task"] == "mtl":
                mle_loss_fn = init_loss_fn(
                    model_config=model_config, state=state, task="mle"
                )
                ooo_loss_fn = init_loss_fn(
                    model_config=model_config,
                    state=state,
                    task="ooo",
                )
                if train:
                    return (mle_loss_fn, ooo_loss_fn)
                return mle_loss_fn
            else:
                loss_fn = init_loss_fn(
                    model_config=model_config, state=state, task="mle"
                )
            return loss_fn

        @partial(jax.jit, static_argnames=["task"])
        def partition_and_merge_grads(
            grads: FrozenDict, state: Any, task: str
        ) -> FrozenDict:
            def get_zero_grads(params: FrozenDict) -> FrozenDict:
                return jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)

            if task == "mle":
                encoder_grads, _ = hk.data_structures.partition(
                    lambda m, n, p: m != "mlp_head", grads
                )
                _, clf_params = hk.data_structures.partition(
                    lambda m, n, p: m != "ooo_head", state.ooo_params
                )
            else:
                encoder_grads, _ = hk.data_structures.partition(
                    lambda m, n, p: m != "ooo_head", grads
                )
                _, clf_params = hk.data_structures.partition(
                    lambda m, n, p: m != "mlp_head", state.mle_params
                )
            zero_grads = get_zero_grads(clf_params)
            grads = hk.data_structures.merge(encoder_grads, zero_grads)
            grads = freeze(grads)
            return grads

        def train_step(
            model_config,
            tasks,
            state,
            batches,
            rng=None,
        ):
            loss_funs = get_loss_fn(
                state,
                model_config,
                train=True if model_config["task"] == "mtl" else False,
            )
            total_loss = 0
            accs = []
            for i, loss_fn in enumerate(loss_funs):
                batch = batches[i]
                # params = partition_and_merge_params(state, tasks[i])
                # get loss, gradients for objective function, and other outputs of loss function
                if model_config["type"].lower() == "resnet":
                    (loss, aux), grads = jax.value_and_grad(
                        loss_fn, argnums=0, has_aux=True
                    )(
                        getattr(state, f"{tasks[i]}_params"),
                        batch,
                        True,
                        model_config["weights"],
                    )
                    # update parameters and batch statistics
                    state = state.apply_gradients(
                        grads=grads,
                        batch_stats=aux[1]["batch_stats"],
                        task=tasks[i],
                    )
                else:
                    if model_config["type"].lower() == "vit":
                        (loss, aux), grads = jax.value_and_grad(
                            loss_fn, argnums=0, has_aux=True
                        )(
                            getattr(state, f"{tasks[i]}_params"),
                            batch,
                            rng,
                            True,
                            model_config["weights"],
                        )
                        self.rng = aux[1]
                    else:
                        (loss, aux), grads = jax.value_and_grad(
                            loss_fn, argnums=0, has_aux=True
                        )(
                            getattr(state, f"{tasks[i]}_params"),
                            batch,
                            model_config["weights"],
                        )
                    # jointly update parameters for both tasks
                    state = state.apply_gradients(grads=grads, task=tasks[i])
                    grads = partition_and_merge_grads(
                        grads=grads, state=state, task=tasks[i]
                    )
                    state = state.apply_gradients(grads=grads, task=tasks[i - 1])

                total_loss += loss
                accs.append(aux)

            # print(f'\nOdd-one-out accuracy: {accs[1]}\n')
            return state, total_loss, accs[0]

        def inference(model_config, state, X, rng=None) -> Array:
            if model_config["type"].lower() == "custom":
                logits = getattr(utils, "cnn_predict")(
                    state=state,
                    params=state.mle_params,
                    X=X,
                    task="mle",
                )
            elif model_config["type"].lower() == "resnet":
                logits = getattr(utils, "resnet_predict")(
                    state=state,
                    params=state.mle_params,
                    X=X,
                    train=False,
                    task="mle",
                )
            elif model_config["type"].lower() == "vit":
                logits, _ = getattr(utils, "vit_predict")(
                    state=state,
                    params=state.mle_params,
                    rng=rng,
                    X=X,
                    train=False,
                    task="mle",
                )
            return logits

        # initialize functions
        self.train_step = partial(train_step, self.model_config, self.tasks)
        self.inference = partial(inference, self.model_config)
        self.get_loss_fn = get_loss_fn

    def eval_step(self, batch: Tuple[Array], cls_hits=None):
        # Return the accuracy for a single batch
        batch = batch[0] if isinstance(batch[0], tuple) else batch
        if hasattr(self, "class_hitting"):
            assert isinstance(
                cls_hits, dict
            ), "\nDictionary to collect class-instance hits required.\n"
            X, y = batch
            logits = self.inference(self.state, X, self.rng)
            loss = optax.softmax_cross_entropy(logits, y).mean()
            batch_hits = getattr(utils, "class_hits")(logits, y)
            acc = self.collect_hits(cls_hits=cls_hits, batch_hits=batch_hits)
        else:
            loss_fn = self.get_loss_fn(self.state, self.model_config, train=False)
            if self.model_config["type"].lower() == "resnet":
                loss, aux = loss_fn(self.state.mle_params, batch, False)
            elif self.model_config["type"].lower() == "vit":
                loss, aux = loss_fn(self.state.mle_params, batch, self.rng, False)
            else:
                loss, aux = loss_fn(self.state.mle_params, batch)
            acc = self.compute_accuracy(batch, aux)
        return loss.item(), acc

    def compute_accuracy(self, batch, aux, cls_hits=None) -> Array:
        logits = aux[0] if isinstance(aux, tuple) else aux
        _, y = batch[0] if isinstance(batch[0], tuple) else batch
        if hasattr(self, "class_hitting"):
            assert isinstance(
                cls_hits, dict
            ), "\nDictionary to collect class-instance hits required.\n"
            batch_hits = getattr(utils, "class_hits")(logits, y)
            acc = self.collect_hits(
                cls_hits=cls_hits,
                batch_hits=batch_hits,
            )
        elif self.model_config["task"] in ["mle", "mtl"]:
            acc = getattr(utils, "accuracy")(logits, y)
        return acc

    @staticmethod
    def collect_hits(
        cls_hits: Dict[int, List[int]], batch_hits: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        for cls, hits in batch_hits.items():
            cls_hits[cls].extend(hits)
        return cls_hits

    def train_epoch(self, batches: Iterator, train: bool) -> Tuple[float, float]:
        """Take a step over each mini-batch in the (train or val) generator."""
        if hasattr(self, "class_hitting"):
            cls_hits = defaultdict(list)
        else:
            batch_accs = jnp.zeros(len(batches))
        batch_losses = jnp.zeros(len(batches))
        # for step, batch in enumerate(batches):
        for step, batch in tqdm(enumerate(batches), desc="Training", leave=False):
            if train:
                self.state, loss, aux = self.train_step(
                    state=self.state, batches=batch, rng=self.rng
                )
                if hasattr(self, "class_hitting"):
                    cls_hits = self.compute_accuracy(
                        batch=batch, aux=aux, cls_hits=cls_hits
                    )
                else:
                    acc = self.compute_accuracy(batch=batch, aux=aux)
                    batch_accs = batch_accs.at[step].set(acc)
            else:
                if hasattr(self, "class_hitting"):
                    loss, cls_hits = self.eval_step(
                        batch=batch,
                        cls_hits=cls_hits,
                    )
                else:
                    loss, acc = self.eval_step(batch)
                    batch_accs = batch_accs.at[step].set(acc)
            batch_losses = batch_losses.at[step].set(loss)
        if hasattr(self, "class_hitting"):
            cls_accs = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
            avg_batch_acc = np.mean(list(cls_accs.values()))
        else:
            avg_batch_acc = jnp.mean(batch_accs)
        avg_batch_loss = jnp.mean(batch_losses)
        return (avg_batch_loss, avg_batch_acc)

    def train(self, train_batches: Iterator, val_batches: Iterator) -> Tuple[dict, int]:
        self.init_optim(train_batches=train_batches)
        for epoch in tqdm(range(1, self.optimizer_config.epochs + 1), desc="Epoch"):
            train_performance = self.train_epoch(batches=train_batches, train=True)
            self.train_metrics.append(train_performance)

            if epoch % 2 == 0:
                test_performance = self.train_epoch(batches=val_batches, train=False)
                self.test_metrics.append(test_performance)
                # Tensorboard logging
                self.logger.add_scalar(
                    "train/loss", np.asarray(train_performance[0]), global_step=epoch
                )
                self.logger.add_scalar(
                    "train/acc", np.asarray(train_performance[1]), global_step=epoch
                )
                self.logger.add_scalar(
                    "val/loss", np.asarray(test_performance[0]), global_step=epoch
                )
                self.logger.add_scalar(
                    "val/acc", np.asarray(test_performance[1]), global_step=epoch
                )
                print(
                    f"Epoch: {epoch:04d}, Train Loss: {train_performance[0]:.4f}, Train Acc: {train_performance[1]:.4f}, Val Loss: {test_performance[0]:.4f}, Val Acc: {test_performance[1]:.4f}\n"
                )
                self.logger.flush()

            if epoch % self.steps == 0:
                self.save_model(epoch=epoch)
                # intermediate_performance = utils.merge_metrics(
                #    (self.train_metrics, self.test_metrics))
                # utils.save_metrics(out_path=os.path.join(
                #    self.out_path, 'metrics'), metrics=intermediate_performance, epoch=f'{epoch+1:04d}')

            if epoch > self.optimizer_config.burnin:
                _, early_stop = self.early_stop.update(test_performance[0])
                if early_stop.should_stop:
                    print("\nMet early stopping criteria, stopping training...\n")
                    break

        metrics = self.merge_metrics(self.train_metrics, self.test_metrics)
        return metrics, epoch

    @staticmethod
    def merge_metrics(train_metrics, test_metrics) -> FrozenDict:
        return flax.core.FrozenDict(
            {"train_metrics": train_metrics, "test_metrics": test_metrics}
        )

    @staticmethod
    def save_metrics(out_path, metrics, epoch):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, f"metrics_{epoch}.pkl"), "wb") as f:
            pickle.dump(metrics, f)

    def save_model(self, epoch=0):
        # Save current model at certain training iteration
        target = {"params": {"mle_params": self.state.mle_params}}
        if self.model_config["task"] == "mtl":
            target["params"].update({"ooo_params": self.state.ooo_params})
        if self.model_config["type"].lower() == "resnet":
            target.update({"params": {"batch_stats": self.state.batch_stats}})
        checkpoints.save_checkpoint(
            ckpt_dir=self.dir_config.log_dir, target=target, step=epoch, overwrite=True
        )

    def load_model(self):
        """Loade model checkpoint. Different checkpoint is used for pretrained models."""
        if self.model_config["type"].lower() == "resnet":
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.dir_config.log_dir, target=None
            )
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                mle_params=state_dict["params"]["mle_params"],
                batch_stats=state_dict["params"]["batch_stats"],
                tx=self.state.tx
                if self.state
                else optax.sgd(self.optimizer_config.lr, momentum=0.9),
                ooo_params=state_dict["params"]["ooo_params"]
                if "ooo_params" in state_dict["params"]
                else None,
            )
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.dir_config.log_dir, target=None
            )
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                mle_params=state_dict["params"]["mle_params"],
                tx=self.state.tx
                if self.state
                else optax.adam(self.optimizer_config.lr),
                batch_stats=None,
                ooo_params=state_dict["params"]["ooo_params"]
                if "ooo_params" in state_dict["params"]
                else None,
            )

    def __len__(self) -> int:
        return self.optimizer_config.epochs
