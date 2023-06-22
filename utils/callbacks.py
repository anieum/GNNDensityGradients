import torch
from visualization import visualize_model_fig, fig_to_tensor
import pytorch_lightning as pl
from pl.callbacks import Callback

# Possible hooks:
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html

# Callsbacks:
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html

class VisualizePredictionCallback(Callback):
    """
    Callback that visualizes the model's prediction on the test dataset.

    :param model: model to visualize
    :param data_loader: data loader to use for visualization
    :param dataset_type: type of the dataset as string ("train", "eval", "test")
    """

    def __init__(self, model, data_loader, dataset_type="train"):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.dataset_type = dataset_type

    def generate_images(self, trainer, model, dataset):
        tensorboard = trainer.logger.experiment

        results = [fig_to_tensor(fig) for fig in visualize_model_fig(model, dataset, same_color_axis=True)]

        for i, result in enumerate(results):
            tensorboard.add_image(f"results/{self.dataset_type}/{i}", result, global_step=trainer.global_step)

    def on_fit_start(self, trainer, pl_module):
        print("Visualizing", self.dataset_type)
        self.generate_images(trainer, pl_module, self.data_loader)

    def on_fit_end(self, trainer, pl_module):
        print("Visualizing", self.dataset_type)
        self.generate_images(trainer, pl_module, self.data_loader)




class ActivationHistogramCallback(Callback):
    def __init__(self, model, data_loader, dataset_type="train"):
        super().__init__()
        self.output_handler = SaveOutputHandler()
        self.ignore_layers = [
            ml3dp.layers.convolutions.ContinuousConv,
            ml3dp.layers.neighbor_search.FixedRadiusSearch,
            ml3dp.layers.neighbor_search.RadiusSearch
        ]
        self.model = model

    def on_fit_start(self, trainer, pl_module):
        # todo: register hook
        # do 1 forward pass with random data to visualize activations
        # write activations to tensorboard
        # unregister hook



# Log activations
GenerateActivationHistogram = False

# Generate activation histogram for each layer
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