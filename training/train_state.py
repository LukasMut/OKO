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
    """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    mle_params: flax.core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    mle_opt_state: optax.OptState
    batch_stats: Any = None
    ooo_params: flax.core.FrozenDict[str, Any] = None
    ooo_opt_state: flax.core.FrozenDict[str, Any] = None
    
    
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
        task = kwargs.pop('task')
        updates, new_opt_state = self.tx.update(
        grads, getattr(self, f'{task}_opt_state'), getattr(self, f'{task}_params'))
        new_params = optax.apply_updates(getattr(self, f'{task}_params'), updates)
        if task == 'mle':
            return self.replace(
                    step=self.step + 1,
                    mle_params=new_params,
                    mle_opt_state=new_opt_state,
                    **kwargs,
                )
        else:
            return self.replace(
                    step=self.step + 1,
                    ooo_params=new_params,
                    ooo_opt_state=new_opt_state,
                    **kwargs,
                )
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        task = kwargs.pop('task')
        if task == 'mtl':
            try:
                ooo_params = kwargs.pop('ooo_params')
                ooo_opt_state = tx.init(ooo_params)
            except KeyError:
                raise Exception('\nTwo sets of params are required for the MTL setting\n')
            mle_opt_state = tx.init(params)
            return cls(
                step=0,
                apply_fn=apply_fn,
                mle_params=params,
                mle_opt_state=mle_opt_state,
                tx=tx,
                ooo_params=ooo_params,
                ooo_opt_state=ooo_opt_state,
                **kwargs,
            )
        else:
            opt_state = tx.init(params)
            return cls(
                step=0,
                apply_fn=apply_fn,
                mle_params=params,
                mle_opt_state=opt_state,
                tx=tx,
                **kwargs,
            ) 
        