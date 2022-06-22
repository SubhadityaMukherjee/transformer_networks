import json
import math
import os
import random
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.set_cmap("cividis")
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")
import matplotlib
from matplotlib.colors import to_rgb

matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sns

sns.reset_orig()
import urllib.request
from urllib.error import HTTPError

import flax
import jax
import jax.numpy as jnp
import optax
import torch
import torch.utils.data as data
import torchvision
from flax import linen as nn
from flax.training import checkpoints, train_state
from jax import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm


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

class TrainerModule:
    def __init__(self, model, CHECKPOINT_PATH, exmp_imgs, lr=1e-3, weight_decay=0.01, seed=42, **model_hparams):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            exmp_imgs - Example imgs, used as input to initialize the model
            lr - Learning rate of the optimizer to use
            weight_decay - Weight decay to use in the optimizer
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)

        self.model = model(**model_hparams)
        self.CHECKPOINT_PATH = CHECKPOINT_PATH

        self.log_dir = os.path.join(self.CHECKPOINT_PATH, "logs")
        self.logger = SummaryWriter(log_dir=self.log_dir)

        self.create_functions()

        self.init_model(exmp_imgs)

    def create_functions(self):
        def calculate_loss(params, rng, batch, train):
            imgs, labels = batch
            labels_onehot = jax.nn.one_hot(
                labels, num_classes=self.model.num_classes)
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply(
                {"params": params},
                imgs,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, rng)

        def train_step(state, rng, batch):
            def loss_fn(params): return calculate_loss(
                params, rng, batch, train=True)

            (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )

            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc

        def eval_step(state, rng, batch):

            _, (acc, rng) = calculate_loss(
                state.params, rng, batch, train=False)
            return rng, acc

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):

        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng}, exmp_imgs, train=True
        )["params"]
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):

        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.lr,
            boundaries_and_scales={
                int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
                int(num_steps_per_epoch * num_epochs * 0.85): 0.1,
            },
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(lr_schedule, weight_decay=self.weight_decay),
        )

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer,
        )

    def train_model(self, train_loader, val_loader, num_epochs=200):
        self.train_loader = train_loader

        self.init_optimizer(num_epochs, len(self.train_loader))

        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar(
                    "val/acc", eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, epoch):

        metrics = defaultdict(list)
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.state, self.rng, loss, acc = self.train_step(
                self.state, self.rng, batch
            )
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar("train/" + key, avg_val, global_step=epoch)

    def eval_model(self, data_loader):

        correct_class, count = 0, 0
        for batch in data_loader:
            self.rng, acc = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):

        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state.params, step=step, overwrite=True
        )

    def load_model(self, name= "ViT.ckpt", pretrained=False):

        if not pretrained:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None)
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(self.CHECKPOINT_PATH, name), target=None
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx if self.state else optax.adamw(self.lr),
        )

    def checkpoint_exists(self, name = "ViT.ckpt"):
        return os.path.isfile(os.path.join(self.CHECKPOINT_PATH, name))

