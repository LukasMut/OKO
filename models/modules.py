import jax.numpy as jnp
import flax.linen as nn

Array = jnp.ndarray


class Sigmoid(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.sigmoid(x)


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x


class Tanh(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.tanh(x)


class Normalization(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x / jnp.linalg.norm(x, ord=2, axis=1)[:, None]
