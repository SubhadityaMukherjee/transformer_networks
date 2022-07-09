from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from jax import random
from flax.training import checkpoints, train_state
import torch
import optax
import jax
import seaborn as sns
from matplotlib.colors import to_rgb
import matplotlib
from IPython.display import set_matplotlib_formats
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from typing import Sequence
from torch.utils import data
import jax.numpy as jnp
import flax

plt.set_cmap("cividis")

set_matplotlib_formats("svg", "pdf")

matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x


def numpy_to_torch(array):
    array = jax.device_get(array)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def write_pickle(obj, fname):
    fname = Path(fname)
    with open(str(fname) + ".pkl", "wb+") as f:
        pickle.dump(obj, f)


def read_pickle(fname):
    fname = Path(fname)
    with open(str(fname) + ".pkl", "rb+") as f:
        return pickle.load(f)


def compute_weight_decay(params):
    """Given a pytree of params, compute the summed $L2$ norm of the params.

    NOTE: For our case with SGD, weight decay ~ L2 regularization. This won't always be the
    case (ex: Adam vs. AdamW).
    """
    param_norm = 0

    weight_decay_params_filter = flax.traverse_util.ModelParamTraversal(
        lambda path, _: ("bias" not in path and "scale" not in path)
    )

    weight_decay_params = weight_decay_params_filter.iterate(params)

    for p in weight_decay_params:
        if p.ndim > 1:
            param_norm += jnp.sum(p**2)

    return param_norm
