import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import open3d.ml.torch as ml3d

from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset


# data
dataset = {
    # Care: there probably is almost no generalization with all datasets being this similar
    "train": VtkDataset("./datasets/data/ParticleData_Fluid_13.vtk"),
    "eval": VtkDataset("./datasets/data/ParticleData_Fluid_14.vtk"),
    "test": VtkDataset("./datasets/data/ParticleData_Fluid_14.vtk")
}


train_loader = DataLoader(dataset["train"], batch_size=32)
val_loader = DataLoader(dataset["eval"], batch_size=32)

# model
model = CConvModel()

# training
trainer = pl.Trainer(num_nodes=1, precision=16, limit_train_batches=0.5, max_epochs=20, log_every_n_steps=1)

print("Starting training")
trainer.fit(model, train_loader, val_loader)
print("Finished training")

# Todo: Tensorboard hook
# Batching