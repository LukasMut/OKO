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
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import training.utils as utils
from training.train_state import TrainState

Array = Any
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
    freeze_encoder: bool = False
    regularization: bool = None

    def __post_init__(self):
        self.rng_seq = hk.PRNGSequence(self.rnd_seed)
        self.rng = jax.random.PRNGKey(self.rnd_seed)
        # inititalize model
        self.init_model()
        # freeze model config dict (i.e., make it immutable)
        self.model_config = FrozenDict(self.model_config)
        self.logger = SummaryWriter(log_dir=self.dir_config.log_dir)
        self.early_stop = EarlyStopping(
            min_delta=1e-4, patience=self.optimizer_config.patience
        )
        # create jitted train and eval functions
        self.create_functions()
        if (
            self.data_config.distribution == "heterogeneous"
            and self.data_config.alpha >= float(0)
            and self.model_config["task"].startswith("mle")
        ):
            self.class_hitting = True

        self.train_metrics = list()
        self.test_metrics = list()

    def init_model(self) -> None:
        """Initialise parameters (i.e., weights and biases) of neural network."""

        @jax.jit
        def init(*args):
            return self.model.init(*args)

        key_i, key_j = random.split(random.PRNGKey(self.rnd_seed))

        if self.data_config.name.endswith("mnist"):
            H, W = self.data_config.input_dim
            C = 1  # gray channel
        else:
            H, W, C = self.data_config.input_dim

        if self.model_config["task"].startswith("ooo"):
            batch = random.normal(
                key_i, shape=(int(self.data_config.batch_size * 3), H, W, C)
            )
        else:
            batch = random.normal(key_i, shape=(self.data_config.batch_size, H, W, C))

        if self.model_config["type"].lower() == "resnet":
            variables = self.model.init(key_j, batch, train=True)
            self.init_params, self.init_batch_stats = (
                variables["params"],
                variables["batch_stats"],
            )
        else:
            if self.model_config["type"].lower() == "vit":
                self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
                self.init_params = self.model.init(
                    {"params": init_rng, "dropout": dropout_init_rng}, batch, train=True
                )["params"]
            else:
                variables = init(key_j, batch)
                _, self.init_params = variables.pop("params")
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
        if self.model_config["fine_tuning"] and self.freeze_encoder:
            self.merge_params()
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats
            if self.state is None
            else self.state.batch_stats,
            tx=optimizer,
            freeze_encoder=self.freeze_encoder,
        )

    def merge_params(self) -> None:
        if self.model_config["pretraining_task"] == "ooo_clf":
            del self.state.params["mlp_head"]
        _, mlp_head = self.init_params.pop("mlp_head")
        self.state.params.update({"mlp_head": mlp_head})

    def create_functions(self) -> None:
        def get_loss_fn(state, model_config: FrozenDict) -> Callable:
            """Get task and model specific loss function."""
            task = (
                "mle"
                if model_config["task"].startswith("mle")
                else model_config["task"]
            )
            if task == "ooo_clf":
                # create all six six permutations
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

        def apply_l2_norm(state: Any, lmbda: float) -> Tuple[Any, Array]:
            weight_penalty, grads = jax.value_and_grad(
                utils.l2_reg, argnums=0, has_aux=False
            )(state.params, lmbda)
            state = state.apply_gradients(grads=grads)
            return state, weight_penalty

        def apply_inductive_bias(
            state: Any,
            pretrained_params: FrozenDict,
            lmbda: float,
        ) -> Tuple[Any, Array]:
            weight_penalty, grads = jax.value_grad(
                utils.inductive_bias, argnums=0, has_aux=False
            )(state.params, pretrained_params, lmbda)
            state = state.apply_gradients(grads=grads)
            return state, weight_penalty

        def train_step(
            model_config,
            state,
            batch,
            rng=None,
            pretrained_params=None,
            inductive_bias=False,
        ):
            loss_fn = get_loss_fn(state, model_config)
            # get loss, gradients for objective function, and other outputs of loss function
            if model_config["type"].lower() == "resnet":
                (loss, aux), grads = jax.value_and_grad(
                    loss_fn, argnums=0, has_aux=True
                )(state.params, batch, True)
                # update parameters and batch statistics
                state = state.apply_gradients(
                    grads=grads, batch_stats=aux[1]["batch_stats"]
                )
            else:
                if model_config["type"].lower() == "vit":
                    (loss, aux), grads = jax.value_and_grad(
                        loss_fn, argnums=0, has_aux=True
                    )(state.params, batch, rng, True)
                    self.rng = aux[1]
                else:
                    (loss, aux), grads = jax.value_and_grad(
                        loss_fn, argnums=0, has_aux=True
                    )(state.params, batch)
                # update parameters
                state = state.apply_gradients(grads=grads)

            if model_config["task"].startswith("ooo"):
                # apply l2 normalization during triplet pretraining
                state, weight_penalty = apply_l2_norm(
                    state=state, lmbda=model_config["weight_decay"]
                )
                loss += weight_penalty

            elif inductive_bias:
                assert not isinstance(
                    pretrained_params, type(None)
                ), "\nTo apply inductive bias during finetuning, pretrained params need to be provided.\n"
                state, weight_penalty = apply_inductive_bias(
                    state=state,
                    pretrained_params=pretrained_params,
                    lmbda=model_config["weight_decay"],
                )
                loss += weight_penalty

            return state, loss, aux

        def inference(model_config, state, X, rng=None) -> Array:
            if model_config["type"].lower() == "custom":
                logits = getattr(utils, "cnn_predict")(state, state.params, X)
            elif model_config["type"].lower() == "resnet":
                logits, _ = getattr(utils, "rn_predict")(
                    state,
                    state.params,
                    X,
                    train=False,
                )
            elif model_config["type"].lower() == "vit":
                logits, _ = getattr(utils, "vit_predict")(
                    state,
                    state.params,
                    rng,
                    X,
                    train=False,
                )
            return logits

        # jit functions for more efficiency
        # self.train_step = jax.jit(train_step)
        # self.eval_step = jax.jit(eval_step)
        self.train_step = partial(train_step, self.model_config)
        self.inference = partial(inference, self.model_config)
        self.get_loss_fn = get_loss_fn

    def eval_step(self, batch: Tuple[Array], cls_hits=None):
        # Return the accuracy for a single batch
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
            loss_fn = self.get_loss_fn(self.state, self.model_config)
            if self.model_config["type"].lower() == "resnet":
                loss, aux = loss_fn(self.state.params, batch, False)
            elif self.model_config["type"].lower() == "vit":
                loss, aux = loss_fn(self.state.params, batch, self.rng, False)
            else:
                loss, aux = loss_fn(self.state.params, batch)
            if self.model_config["task"] == "ooo_clf":
                acc = aux[0] if isinstance(aux, tuple) else aux
            else:
                acc = self.compute_accuracy(batch, aux)
        return loss.item(), acc

    def compute_accuracy(self, batch, aux, cls_hits=None) -> Array:
        logits = aux[0] if isinstance(aux, tuple) else aux
        _, y = batch
        if hasattr(self, "class_hitting"):
            assert isinstance(
                cls_hits, dict
            ), "\nDictionary to collect class-instance hits required.\n"
            batch_hits = getattr(utils, "class_hits")(logits, y)
            acc = self.collect_hits(
                cls_hits=cls_hits,
                batch_hits=batch_hits,
            )
        elif self.model_config["task"] == "ooo_dist":
            acc = getattr(utils, "ooo_accuracy")(logits, y)
        elif self.model_config["task"].startswith("mle"):
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
        # for step, batch in tqdm(enumerate(batches), desc="Training", leave=False):
        for step, batch in enumerate(batches):
            if train:
                self.state, loss, aux = self.train_step(
                    state=self.state,
                    batch=batch,
                    rng=self.rng,
                    pretrained_params=self.pretrained_params
                    if hasattr(self, "pretrained_params")
                    else None,
                    inductive_bias=hasattr(self, "inductive_bias"),
                )
                if self.model_config["task"] == "ooo_clf":
                    acc = aux[0] if isinstance(aux, tuple) else aux
                    batch_accs = batch_accs.at[step].set(acc)
                else:
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
        if self.model_config["task"].endswith("finetuning") and not self.freeze_encoder:
            setattr(self, "inductive_bias", True)
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
        if self.model_config["type"].lower() == "resnet":
            target = {
                "params": self.state.params,
                "batch_stats": self.state.batch_stats,
            }
        else:
            target = self.state.params
        checkpoints.save_checkpoint(
            ckpt_dir=self.dir_config.log_dir, target=target, step=epoch, overwrite=True
        )

    def load_model(self, pretrained=False):
        """Loade model checkpoint. Different checkpoint is used for pretrained models."""
        if self.model_config["type"].lower() == "resnet":
            if not pretrained:
                state_dict = checkpoints.restore_checkpoint(
                    ckpt_dir=self.dir_config.log_dir, target=None
                )
            else:
                state_dict = checkpoints.restore_checkpoint(
                    ckpt_dir=self.dir_config.pretraining_dir,
                    target=None,
                )

            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=state_dict["params"],
                batch_stats=state_dict["batch_stats"],
                tx=self.state.tx
                if self.state
                else optax.sgd(self.optimizer_config.lr, momentum=0.9),
                freeze_encoder=False,
            )
            if self.model_config["task"].endswith("finetuning"):
                setattr(self, "pretrained_params", state_dict["params"])
        else:
            if not pretrained:
                params = checkpoints.restore_checkpoint(
                    ckpt_dir=self.dir_config.log_dir, target=None
                )
            else:
                params = checkpoints.restore_checkpoint(
                    ckpt_dir=self.dir_config.pretraining_dir,
                    target=None,
                )

            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.state.tx
                if self.state
                else optax.adam(self.optimizer_config.lr),  # default ViT optimizer
                freeze_encoder=False,
                batch_stats=None,
            )

            if self.model_config["task"].endswith("finetuning"):
                setattr(self, "pretrained_params", params)

    def __len__(self) -> int:
        return self.optimizer_config.epochs
