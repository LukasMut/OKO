#!/usr/bin/env python3
# -*- coding: utf-8 -*

from typing import Any, Callable
from flax import struct
from dataclasses import dataclass

import flax
import optax

@dataclass
class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    source: https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html#TrainState
    
    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
        step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
        batch_stats: storing batch norm stats
        freeze_encoder: if True only updates parameters of linear probe / classification head
    """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    freeze_encoder: bool
    batch_stats: Any = None
    
    
    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        if self.freeze_encoder:
            head_grads = grads["mlp_head"]
            head_params = self.params["mlp_head"]
            updates, new_opt_state = self.tx.update(
                    head_grads, self.opt_state, head_params
            )
            updated_head = optax.apply_updates(head_params, updates)
            del self.params["mlp_head"]
            self.params.update({"mlp_head": updated_head})
            new_params = self.params
        else:
            updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)

        return self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                **kwargs,
            )
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, freeze_encoder, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params["mlp_head"]) if freeze_encoder else tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )