__all__ = ["ColorAugmentations", "RandomCrop", "Resize"]

from dataclasses import dataclass
from typing import Iterator, Tuple, Union

import dm_pix as pix
import jax
import numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jaxtyping import AbstractDtype, Array, Float32
from ml_collections import config_dict

FrozenDict = config_dict.FrozenConfigDict


class FP32or64(AbstractDtype):
    dtypes = ["float32", "float64"]


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class ColorAugmentations:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    random_numbers: Iterator

    def tree_flatten(self):
        children = ()
        aux_data = {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
            "random_numbers": self.random_numbers,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def get_key(self) -> Array:
        return jax.random.PRNGKey(next(self.random_numbers))

    def _jitter_brightness(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]]:
        """Jitter brightness of batch of images."""
        key = self.get_key()
        keys = jax.random.split(key, num=batch.shape[0])
        batch = vmap(
            lambda k, x: pix.random_brightness(
                key=k, image=x, max_delta=self.brightness
            )
        )(keys, batch)
        return batch

    def _jitter_contrast(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]]:
        """Jitter contrast of batch of images."""
        key = self.get_key()
        keys = jax.random.split(key, num=batch.shape[0])
        batch = vmap(
            lambda k, x: pix.random_contrast(
                key=k, image=x, lower=1 - self.contrast, upper=1 + self.contrast
            )
        )(keys, batch)
        return batch

    def _jitter_saturation(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]]:
        """Jitter saturation of batch of images."""
        key = self.get_key()
        keys = jax.random.split(key, num=batch.shape[0])
        batch = vmap(
            lambda k, x: pix.random_saturation(
                key=k, image=x, lower=1 - self.saturation, upper=1 + self.saturation
            )
        )(keys, batch)
        return batch

    def _jitter_hue(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]]:
        """Jitter hue of batch of images."""
        key = self.get_key()
        keys = jax.random.split(key, num=batch.shape[0])
        batch = vmap(lambda k, x: pix.random_hue(key=k, image=x, max_delta=self.hue))(
            keys, batch
        )
        return batch

    @jax.jit
    def apply(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ):
        """Apply color augmentations to a batch of images."""
        batch = self._jitter_brightness(batch)
        batch = self._jitter_contrast(batch)
        batch = self._jitter_saturation(batch)
        batch = self._jitter_hue(batch)
        return batch


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class RandomCrop:
    crop_size: Tuple[int, int, int]
    random_numbers: Iterator

    def tree_flatten(self):
        children = ()
        aux_data = {
            "crop_size": self.crop_size,
            "random_numbers": self.random_numbers,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def get_key(self) -> Array:
        return jax.random.PRNGKey(next(self.random_numbers))

    def random_crop(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[
        FP32or64[np.ndarray, "#batch h_prime w_prime c"],
        Float32[Array, "#batch h_prime w_prime c"],
    ]:
        """Jitter hue of batch of images."""
        key = self.get_key()
        keys = jax.random.split(key, num=batch.shape[0])
        batch = vmap(
            lambda k, x: pix.random_crop(key=k, image=x, crop_sizes=self.crop_size)
        )(keys, batch)
        return batch

    @jax.jit
    def apply(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[
        FP32or64[np.ndarray, "#batch h_prime w_prime c"],
        Float32[Array, "#batch h_prime w_prime c"],
    ]:
        """Apply color augmentations to a batch of images."""
        return self.random_crop(batch)


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class Resize:
    crop_size: Tuple[int, int, int]

    def tree_flatten(self):
        children = ()
        aux_data = {
            "crop_size": self.crop_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def resize(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[
        FP32or64[np.ndarray, "#batch h_prime w_prime c"],
        Float32[Array, "#batch h_prime w_prime c"],
    ]:
        """Jitter hue of batch of images."""
        return vmap(
            lambda x: jax.image.resize(
                x, shape=self.crop_size, method="bilinear", antialias=True
            )
        )(batch)

    @jax.jit
    def apply(
        self,
        batch: Union[
            FP32or64[np.ndarray, "#batch h w c"], Float32[Array, "#batch h w c"]
        ],
    ) -> Union[
        FP32or64[np.ndarray, "#batch h_prime w_prime c"],
        Float32[Array, "#batch h_prime w_prime c"],
    ]:
        """Apply color augmentations to a batch of images."""
        return self.resize(batch)
