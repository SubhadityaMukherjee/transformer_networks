# Mostly from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
# %%
# LOADING EVERYTHING
from models import *
from blocks import *
from utils import *
from torchvision.datasets import *
from torchvision import transforms
from jax import random
import torchvision
import torch.utils.data as data
import torch
import jax
import seaborn as sns
from matplotlib.colors import to_rgb
import matplotlib
from IPython.display import set_matplotlib_formats
import os

import matplotlib.pyplot as plt
import numpy as np

plt.set_cmap("cividis")

set_matplotlib_formats("svg", "pdf")

matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()


# %%
# SETTING UP DEFAULTS
DATASET_PATH = "/media/hdd/Datasets"
CHECKPOINT_PATH = "saved_models/viTJax"

main_rng = random.PRNGKey(42)

print("Device:", jax.devices()[0])
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
# %%

# DATA TRANSFORMS
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - DATA_MEANS) / DATA_STD
    return img


# %%
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
train_set, _ = torch.utils.data.random_split(
    train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
)
_, val_set = torch.utils.data.random_split(
    val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
)

test_set = ds_name(
    root=DATASET_PATH, train=False, transform=test_transform, download=True
)
# %%
# DATA LOADERS
batch_size = 128

train_loader = data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=numpy_collate,
    num_workers=8,
    persistent_workers=True,
)
val_loader = data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
    num_workers=4,
    persistent_workers=True,
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
# %%
# VISUALIZING THINGS
NUM_IMAGES = 4
CIFAR_images = np.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], axis=0)
img_grid = torchvision.utils.make_grid(
    numpy_to_torch(CIFAR_images), nrow=4, normalize=True, pad_value=0.9
)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(8, 8))
plt.title("Image examples of the CIFAR10 dataset")
plt.imshow(img_grid)
plt.axis("off")
plt.savefig("outputs/image-examples.png")
plt.close()
# %%

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

plt.savefig("outputs/patches.png")
plt.close()

# %%
<<<<<<< HEAD

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 128))
attnblock = AttentionBlock(embed_dim=128, hidden_dim=512, num_heads=4, dropout_prob=0.1)
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = attnblock.init({"params": init_rng, "dropout": dropout_init_rng}, x, True)[
    "params"
]
main_rng, dropout_apply_rng = random.split(main_rng)
out = attnblock.apply(
    {"params": params}, x, train=True, rngs={"dropout": dropout_apply_rng}
)
print("Out", out.shape)

del attnblock, params
#%%

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (5, 32, 32, 3))
visntrans = VisionTransformer(
    embed_dim=128,
    hidden_dim=512,
    num_heads=4,
    num_channels=3,
    num_layers=6,
    num_classes=10,
    patch_size=4,
    num_patches=64,
    dropout_prob=0.1,
)
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = visntrans.init({"params": init_rng, "dropout": dropout_init_rng}, x, True)[
    "params"
]
main_rng, dropout_apply_rng = random.split(main_rng)
out = visntrans.apply(
    {"params": params}, x, train=True, rngs={"dropout": dropout_apply_rng}
)
print("Out", out.shape)

del visntrans, params
#%%


=======
# ACTUAL TRAINING
>>>>>>> 8f4c236 (up)
def train_model(*args, num_epochs=200, retrain=False, **kwargs):
    trainer = TrainerModule(*args, **kwargs)
    if not trainer.checkpoint_exists() or retrain == True:
        print("Training")
        trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
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
    num_epochs=2,
    CHECKPOINT_PATH=CHECKPOINT_PATH,
    model=VisionTransformer,
)
print("ViT results", results)
