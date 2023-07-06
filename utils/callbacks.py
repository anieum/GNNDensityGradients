import torch
import open3d.ml.torch.python as ml3dp
from utils.visualization import visualize_model_fig, fig_to_tensor, SaveOutputHandler, plot_sample_for_tensorboard

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Possible hooks:
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html

# Callsbacks:
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html

class VisualizePredictionCallback(Callback):
    """
    Callback that visualizes the model's prediction on the test dataset.

    :param model: model to visualize
    :param data_loader: data loader to use for visualization
    :param dataset_type: type of the dataset as string ("train", "val", "test")
    """

    def __init__(self, model, dataset, dataset_type="train"):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.dataset_type = dataset_type

    def generate_images(self, trainer, model, dataset):
        tensorboard = trainer.logger.experiment

        title = self.dataset_type.capitalize() + " dataset sample"
        result = fig_to_tensor(visualize_model_fig(model, dataset, same_color_axis=True, title=title))
        tensorboard.add_image(f"results/{self.dataset_type}", result, global_step=trainer.global_step)

    def on_fit_start(self, trainer, pl_module):
        print("Visualizing", self.dataset_type)
        self.generate_images(trainer, pl_module, self.dataset)

    def on_fit_end(self, trainer, pl_module):
        print("Visualizing", self.dataset_type)
        self.generate_images(trainer, pl_module, self.dataset)


class VisulizePrediction3DCallback(Callback):

    def __init__(self, model, dataset, dataset_type="train"):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.dataset_type = dataset_type

    def generate_images(self, trainer, model, dataset):
        tensorboard = trainer.logger.experiment

        result = fig_to_tensor(plot_sample_for_tensorboard(model, dataset))
        tensorboard.add_image(f"results/{self.dataset_type}_3d", result, global_step=trainer.global_step)

    def on_fit_start(self, trainer, pl_module):
        # Todo: Once this works, only on fit end
        print("Visualizing", self.dataset_type, "3D")
        self.generate_images(trainer, pl_module, self.dataset)

    def on_fit_end(self, trainer, pl_module):
        print("Visualizing", self.dataset_type, "3D")
        self.generate_images(trainer, pl_module, self.dataset)


class ActivationHistogramCallback(Callback):
    """
    Callback that generates a histogram of the activations of the model for each layer
    and logs it to tensorboard.

    :param model: model for which the activations of each layer should be generated
    """

    def __init__(self, model):
        super().__init__()
        self.hook_handles = []
        self.ignore_layers = [
            ml3dp.layers.convolutions.ContinuousConv,
            ml3dp.layers.neighbor_search.FixedRadiusSearch,
            ml3dp.layers.neighbor_search.RadiusSearch
        ]
        self.model = model

    def generate_activations(self, trainer):
        tensorboard = trainer.logger.experiment
        data_saver = SaveOutputHandler()


        # register hook that stores activations
        for layer in self.model.modules():
            if any(isinstance(layer, L) for L in self.ignore_layers):
                continue

            handle = layer.register_forward_hook(data_saver)
            self.hook_handles.append(handle)

        # run model and log activations
        features = torch.rand(1000, 4)
        particle_positions = torch.rand(1000, 3)
        input = ([features], [particle_positions], particle_positions)
        self.model(input)

        # write activations to tensorboard
        for i, x in enumerate(data_saver.outputs):
            tensorboard.add_histogram("Initial activations (no o3d layers)", x.detach().numpy(), i+1)

        # clear stored data and remove hooks
        data_saver.clear()
        for handle in self.hook_handles:
            handle.remove()

    def on_fit_start(self, trainer, pl_module):
        self.generate_activations(trainer)
