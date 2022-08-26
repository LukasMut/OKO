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


class AttentionBlock(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim),
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    @nn.compact
    def __call__(self, x, train=True) -> Array:
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        return x
