import matplotlib.pyplot as plt

plt.set_cmap("cividis")
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")
import matplotlib
from matplotlib.colors import to_rgb

matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sns

sns.reset_orig()

from flax import linen as nn

from utils import *


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
