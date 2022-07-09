import jax.numpy as jnp
from typing import *
from .utils import *
from flax import linen as nn
import seaborn as sns
from matplotlib.colors import to_rgb
import matplotlib
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt

plt.set_cmap("cividis")

set_matplotlib_formats("svg", "pdf")

matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()


class AttentionBlock(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    dropout_prob: float = 0.0

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

    def __call__(self, x, train=True):
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


class VisionTransformer(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_channels: int
    num_layers: int
    num_classes: int
    patch_size: int
    num_patches: int
    dropout_prob: float = 0.0

    def setup(self):

        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [
            AttentionBlock(
                self.embed_dim, self.hidden_dim, self.num_heads, self.dropout_prob
            )
            for _ in range(self.num_layers)
        ]
        self.mlp_head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])
        self.dropout = nn.Dropout(self.dropout_prob)

        self.cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim)
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + self.num_patches, self.embed_dim),
        )

    def __call__(self, x, train=True):

        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:, : T + 1]

        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)

        cls = x[:, 0]
        out = self.mlp_head(cls)
        return out
