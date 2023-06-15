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
from utils.visualization import visualize_model_fig
from utils.visualization import fig_to_tensor


# data
dataset = {
    # Care: there probably is almost no generalization with all datasets being this similar
    "train": VtkDataset("./datasets/data/ParticleData_Fluid_13.vtk"),
    "eval": VtkDataset("./datasets/data/ParticleData_Fluid_14.vtk"),
    "test": VtkDataset("./datasets/data/ParticleData_Fluid_14.vtk")
}


train_loader = DataLoader(dataset["train"], batch_size=841)
val_loader = DataLoader(dataset["eval"], batch_size=841)

# model
model = CConvModel()

# training
# see https://lightning.ai/docs/pytorch/stable/common/trainer.html
trainer = pl.Trainer(num_nodes=1, precision=16, max_epochs=600, log_every_n_steps=1, check_val_every_n_epoch=10)

print("Starting training")
trainer.fit(model, train_loader, val_loader)
print("Finished training")

# Todo: do this via hook
for dataset_type in "train", "eval":
    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('lightning_logs')

    print("Visualizing", dataset_type)
    results = [fig_to_tensor(fig) for fig in visualize_model_fig(model, dataset["train"], same_color_axis=True)]
    # trainer.logger.experiment.add_images(f"results/{dataset_type}", torch.stack(results), global_step=trainer.global_step)
    
    for i, result in enumerate(results):
        trainer.logger.experiment.add_image(f"results/{dataset_type}/{i}", result, global_step=trainer.global_step)
        # writer.add_image("Final train dataset performance", result)

# evaluation
# trainer.test(model, test_dataloaders=test_loader)
