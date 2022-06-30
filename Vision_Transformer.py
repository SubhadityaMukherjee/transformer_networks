# +
# Mostly from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
# +
# LOADING EVERYTHING
import os
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision
from IPython.display import set_matplotlib_formats
from jax import random
from matplotlib.colors import to_rgb
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import *
from tqdm.notebook import tqdm

from src.blocks import *
from src.models import *
from src.utils import *

plt.set_cmap("cividis")

set_matplotlib_formats("svg", "pdf")

matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()


# +
# SETTING UP DEFAULTS
DATASET_PATH = "/media/hdd/Datasets"
CHECKPOINT_PATH = "saved_models/viTJax"
valid_size = 0.2
main_rng = random.PRNGKey(42)

print("Device:", jax.devices()[0])
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])


# -
# DATA TRANSFORMS
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - DATA_MEANS) / DATA_STD
    return img


test_transform = image_to_numpy
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        image_to_numpy,
    ]
)
ds_name = CIFAR10
train_dataset = ds_name(
    root=DATASET_PATH, train=True, transform=train_transform, download=True
)
val_dataset = ds_name(
    root=DATASET_PATH, train=True, transform=test_transform, download=True
)
# +
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

# train_set, _ = torch.utils.data.random_split(
#     train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42),
# )
# _, val_set = torch.utils.data.random_split(
#     val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42),
# )

test_set = ds_name(
    root=DATASET_PATH, train=False, transform=test_transform, download=True
)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
# +
# DATA LOADERS
batch_size = 128

train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    drop_last=True,
    collate_fn=numpy_collate,
    num_workers=8,
    persistent_workers=True,
    sampler=train_sampler,
)
val_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    drop_last=False,
    collate_fn=numpy_collate,
    num_workers=4,
    persistent_workers=True,
    sampler=valid_sampler,
)
test_loader = data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
    num_workers=4,
    persistent_workers=True,
)
# +
# VISUALIZING THINGS
if not Path.exists(Path("outputs")):
    os.mkdir("outputs")
NUM_IMAGES = 4
CIFAR_images = np.stack([test_set[idx][0] for idx in range(NUM_IMAGES)], axis=0)
img_grid = torchvision.utils.make_grid(
    numpy_to_torch(CIFAR_images), nrow=4, normalize=True, pad_value=0.9
)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(8, 8))
plt.title("Image examples of the CIFAR10 dataset")
plt.imshow(img_grid)
plt.axis("off")
plt.savefig("outputs/ViT-image-examples.png", dpi=200)
plt.close()
# +
img_patches = img_to_patch(CIFAR_images, patch_size=4, flatten_channels=False)

fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14, 3))
fig.suptitle("Images as input sequences of patches")
for i in range(CIFAR_images.shape[0]):
    img_grid = torchvision.utils.make_grid(
        numpy_to_torch(img_patches[i]), nrow=64, normalize=True, pad_value=0.9
    )
    img_grid = img_grid.permute(1, 2, 0)
    ax[i].imshow(img_grid)
    ax[i].axis("off")
#  plt.show()

plt.savefig("outputs/ViT-patches.png", dpi=200)
plt.close()


# -


class TrainerModule:
    def __init__(
        self,
        model,
        CHECKPOINT_PATH,
        exmp_imgs,
        lr=1e-3,
        weight_decay=0.01,
        seed=42,
        **model_hparams
    ):
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
        self.loss_log = []
        self.metric_log = []

    def create_functions(self):
        def calculate_loss(params, rng, batch, train):
            imgs, labels = batch
            labels_onehot = jax.nn.one_hot(labels, num_classes=self.model.num_classes)
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
            def loss_fn(params):
                return calculate_loss(params, rng, batch, train=True)

            (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )

            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc

        def eval_step(state, rng, batch):

            _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
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

    def train_model(
        self, train_loader, val_loader, num_epochs=200, graph_progress=None
    ):
        self.train_loader = train_loader

        self.init_optimizer(num_epochs, len(self.train_loader))

        best_eval = 0.0
        tq = tqdm(range(1, num_epochs + 1))
        for epoch_idx in tq:
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar("val/acc", eval_acc, global_step=epoch_idx)
                tq.set_postfix({"val/acc": eval_acc})
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

    def load_model(self, name="ViT.ckpt", pretrained=False):

        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(self.CHECKPOINT_PATH, name), target=None
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx if self.state else optax.adamw(self.lr),
        )

    def checkpoint_exists(self, name="ViT.ckpt"):
        return os.path.isfile(os.path.join(self.CHECKPOINT_PATH, name))


# ACTUAL TRAINING
def train_model(*args, num_epochs=200, retrain=False, graph_progress=None, **kwargs):
    trainer = TrainerModule(*args, **kwargs)
    if not trainer.checkpoint_exists() or retrain == True:
        print("Training")
        trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            graph_progress=graph_progress,
        )
        trainer.load_model()
    else:
        print("Skipping training")
        trainer.load_model(pretrained=True)
    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    return trainer, {"val": val_acc, "test": test_acc}


model, results = train_model(
    exmp_imgs=next(iter(train_loader))[0],
    embed_dim=256,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    patch_size=4,
    num_channels=3,
    num_patches=64,
    num_classes=10,
    dropout_prob=0.2,
    lr=3e-4,
    retrain=True,
    num_epochs=10,
    CHECKPOINT_PATH=CHECKPOINT_PATH,
    model=VisionTransformer,
    graph_progress=10,
)
print("ViT results", results)
