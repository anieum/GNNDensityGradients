import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import open3d.ml.torch as ml3d
import open3d.ml.torch.python as ml3dp

from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset
from utils.visualization import visualize_model_fig
from utils.visualization import fig_to_tensor
from utils.visualization import SaveOutputHandler

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
trainer = pl.Trainer(num_nodes=1, precision=16, max_epochs=100, log_every_n_steps=1, check_val_every_n_epoch=10)

print("Starting training")
trainer.fit(model, train_loader, val_loader)
print("Finished training")

GenerateImages = True

# Todo: do this via hook
if GenerateImages:
    for dataset_type in "train", "eval":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter('lightning_logs')

        print("Visualizing", dataset_type)
        results = [fig_to_tensor(fig) for fig in visualize_model_fig(model, dataset[dataset_type], same_color_axis=True)]
        # trainer.logger.experiment.add_images(f"results/{dataset_type}", torch.stack(results), global_step=trainer.global_step)
        
        for i, result in enumerate(results):
            trainer.logger.experiment.add_image(f"results/{dataset_type}/{i}", result, global_step=trainer.global_step)
            # writer.add_image("Final train dataset performance", result)

# Log activations
GenerateActivationHistogram = True

# import numpy as np
# for i in range(10):
#     x = np.random.random(1000)
#     trainer.logger.experiment.add_histogram('distribution centers', x + i, i)


# Generate activation histogram for each layer
# Todo: This probably should be before training
if GenerateActivationHistogram:
    hook_handles = []
    save_output = SaveOutputHandler()

    # register hook for each layer
    for layer in model.modules():
        ignore_layers = [ml3dp.layers.convolutions.ContinuousConv, ml3dp.layers.neighbor_search.FixedRadiusSearch, ml3dp.layers.neighbor_search.RadiusSearch]
    
        if any(isinstance(layer, L) for L in ignore_layers):
            continue
    
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
        
    # run model and log activations

    features = torch.rand(1000, 4)
    particle_positions = torch.rand(1000, 3)
    input = ([features], [particle_positions], particle_positions)
    output = model(input)

    # write activations to tensorboard
    for i, x in enumerate(save_output.outputs):
        # writer.add_histogram("layer (exclude o3d layers)", x, i+1)
        trainer.logger.experiment.add_histogram("Initial activations (no o3d layers)", x.detach().numpy(), i+1)

    save_output.clear()

# evaluation
# trainer.test(model, test_dataloaders=test_loader)

# see https://github.com/wi-re/torchSPHv2/blob/master/Cconv/1D/Untitled.ipynb