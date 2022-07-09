# +
# Mostly from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
# +
# LOADING EVERYTHING

# specific snippet to load the networks
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from pathlib import Path
from IPython.display import set_matplotlib_formats
import matplotlib
from matplotlib.colors import to_rgb
import seaborn as sns
import jax
import flax
import torch
import optax
import torch.utils.data as data
import torchvision
from jax import random
from torchvision import transforms
from torchvision.datasets import *
from blocks.utils import *
from blocks.convnext import *
from PIL import Image
import argparse as ap


plt.set_cmap("cividis")

set_matplotlib_formats("svg", "pdf")

matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()

# Argument setup
ag = ap.ArgumentParser()
ag.add_argument("-e", type=int, default=1, help="No of epochs")
ag.add_argument("-bs", type=int, default=64, help="Batch size")
ag.add_argument("-lr", type=float, default=3e-4, help="No of epochs")
ag.add_argument("-load", action="store_true", help="Load saved objects")
args = ag.parse_args()

# +
# SETTING UP DEFAULTS
DATASET_PATH = "/media/hdd/Datasets"
CHECKPOINT_PATH = "saved_models/lent"
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
if not args.load:

    train_dataset = ds_name(
        root=DATASET_PATH, train=True, transform=train_transform, download=True
    )
    write_pickle(train_dataset, "trainds")
    val_dataset = ds_name(
        root=DATASET_PATH, train=True, transform=test_transform, download=True
    )
    write_pickle(val_dataset, "valds")
else:
    train_dataset = read_pickle("trainds")
    val_dataset = read_pickle("valds")
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
batch_size = args.bs

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
#  if not Path.exists(Path("outputs")):
#      os.mkdir("outputs")
#  NUM_IMAGES = 4
#  CIFAR_images = np.stack([test_set[idx][0]
#                          for idx in range(NUM_IMAGES)], axis=0)
#  img_grid = torchvision.utils.make_grid(
#      numpy_to_torch(CIFAR_images), nrow=4, normalize=True, pad_value=0.9
#  )
#  img_grid = img_grid.permute(1, 2, 0)
#
#  plt.figure(figsize=(8, 8))
#  plt.title("Image examples of the CIFAR10 dataset")
#  plt.imshow(img_grid)
#  plt.axis("off")
#  plt.savefig("outputs/ViT-image-examples.png", dpi=200)
#  plt.close()
#
#  img_patches = img_to_patch(CIFAR_images, patch_size=4, flatten_channels=False)
#
#  fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14, 3))
#  fig.suptitle("Images as input sequences of patches")
#  for i in range(CIFAR_images.shape[0]):
#      img_grid = torchvision.utils.make_grid(
#          numpy_to_torch(img_patches[i]), nrow=64, normalize=True, pad_value=0.9
#      )
#      img_grid = img_grid.permute(1, 2, 0)
#      ax[i].imshow(img_grid)
#      ax[i].axis("off")
#
#  plt.savefig("outputs/ViT-patches.png", dpi=200)
#  plt.close()
#

# -


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any


class TrainerModule:
    def __init__(
        self,
        exmp_imgs,
        lr=1e-3,
        weight_decay=0.01,
        seed=42,
        num_epochs=1,
        model=None,
        **model_hparams,
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
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        # Create empty model. Note: no parameters yet
        self.model = model(**model_hparams)
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, "lenet/")
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch_stats, batch, train):
            imgs, labels = batch
            labels_onehot = jax.nn.one_hot(labels, num_classes=self.model.num_classes)
            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply(
                {"params": params, "batch_stats": batch_stats},
                imgs,
                train=train,
                mutable=["batch_stats"] if train else False,
            )
            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, new_model_state)

        # Training function

        def train_step(state, batch):
            def loss_fn(params):
                return calculate_loss(params, state.batch_stats, batch, train=True)

            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state["batch_stats"]
            )
            return state, loss, acc

        # Eval function

        def eval_step(state, batch):
            # Return the accuracy for a single batch
            _, (acc, _) = calculate_loss(
                state.params, state.batch_stats, batch, train=False
            )
            return acc

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = (
            variables["params"],
            variables["batch_stats"],
        )
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
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
        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats
            if self.state is None
            else self.state.batch_stats,
            tx=optimizer,
        )

    def train_model(self, train_loader, val_loader):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(self.num_epochs, len(train_loader))
        # Track best eval accuracy
        best_eval = 0.0
        bar = tqdm(range(1, self.num_epochs + 1))
        for epoch_idx in bar:
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar("val/acc", eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()
                bar.set_description(f"Val/acc: {eval_acc}")

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in bar:
            self.state, loss, acc = self.train_step(self.state, batch)
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
            bar.set_description(f"loss : {loss} , acc : {acc}")
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar("train/" + key, avg_val, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target={"params": self.state.params, "batch_stats": self.state.batch_stats},
            step=step,
            overwrite=True,
        )

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None
            )
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(CHECKPOINT_PATH, f"{self.model_name}.ckpt"),
                target=None,
            )
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict["params"],
            batch_stats=state_dict["batch_stats"],
            tx=self.state.tx if self.state else optax.sgd(0.1),  # Default optimizer
        )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"ViT.ckpt"))


def train_model(*args, num_epochs=1, **kwargs):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(*args, num_epochs=num_epochs, **kwargs)
    trainer.train_model(train_loader, val_loader)
    # Test trained model
    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    return trainer, {"val": val_acc, "test": test_acc}


model, results = train_model(
    exmp_imgs=next(iter(train_loader))[0],
    num_classes=10,
    lr=args.lr,
    num_epochs=args.e,
    model=ResNet32,
)
print("ViT results", results)

imsize = 256
loader = transforms.Compose([transforms.Resize(imsize)])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = jax.numpy.array(loader(image))
    return image  # assumes that you're using GPU


test_im = image_loader(
    "/media/hdd/Datasets/boat/cruise ship/adventure-of-the-seas-cruise-ship-caribb-1218316.jpg"
)
