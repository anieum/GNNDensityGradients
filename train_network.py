from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset


# data
dataset = {
    # Care: there probably is almost no generalization with all datasets being this similar
    "train": VtkDataset("dambreak_2d_test/ParticleData_Fluid_10.vtk"),
    "eval": VtkDataset("dambreak_2d_test/ParticleData_Fluid_11.vtk"),
    "test": VtkDataset("dambreak_2d_test/ParticleData_Fluid_12.vtk")
}


train_loader = DataLoader(dataset["train"], batch_size=32)
val_loader = DataLoader(dataset["eval"], batch_size=32)

# model
model = CConvModel()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)


# Todo: Tensorboard hook
# Batching