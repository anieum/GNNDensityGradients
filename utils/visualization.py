import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from torch.nn.functional import mse_loss
import matplotlib


def plot_particles(positions, color=None, colorscale = 'Viridis'):
    """
    Plots the positions in a 3D scatter plot.
    """
    marker = dict(
        size=5,
        colorscale=colorscale,
        opacity=0.8,
    )

    if type(positions) == torch.Tensor:
        positions = positions.cpu()

    if color is not None:
        if type(color) == np.ndarray:
            marker['color'] = color.reshape(-1)
        elif type(color) == torch.Tensor:
            marker['color'] = color.cpu().view(-1)
        else:
            marker['color'] = color

    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 2],
        z=positions[:, 1],
        mode='markers',
        marker=marker
    )])

    fig.show()


# Build grid with all particles
def get_grid(vector):
    """
    Builds a grid from a vector. The values of the grid are filled row-wise.
    """
    vector = vector.squeeze()
    size = int(np.ceil(np.sqrt(len(vector))))
    grid = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            idx = i * size + j
            if idx < len(vector):
                grid[i][j] = vector[idx]

    return grid

def visualize_dataset(dataset, same_color_axis = False, title = "Normalized Density vs. Normalized Density Gradient"):
    raise NotImplementedError("This function wasn't updated to the new dataset format")

    fig = make_subplots(rows=1, cols=2)
    grid1 = get_grid(dataset.data)
    grid2 = get_grid(dataset.target)

    if same_color_axis:
        fig.add_trace(go.Heatmap(z=grid1, coloraxis = "coloraxis"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=grid2, coloraxis = "coloraxis"), row=1, col=2)
        fig.update_layout(coloraxis = {'colorscale':'plasma'})
    else:
        fig.add_trace(go.Heatmap(z=grid1), row=1, col=1)
        fig.add_trace(go.Heatmap(z=grid2), row=1, col=2)

    fig.update_layout(title_text=title)

    fig.show()

# todo: move this into the dataset class
def transform_sample(sample, transform):
    """
    Applies a transform to a single sample.
    """
    for key in sample:
        if type(sample[key]) == torch.Tensor:
            sample[key] = transform(sample[key])

    return sample

def transform_batch(batch, transform):
    """
    Applies a transform to a batch of samples.
    """
    for i in range(len(batch)):
        batch[i] = transform_sample(batch[i], transform)

    return batch


def visualize_model_fig(model,
                    dataset,
                    same_color_axis = False,
                    title = None,
                    subplot_titles = ('Densities',  'Target density grad', 'Predicted density grad'),
                    width = 1000,
                    height = 450):
    # Pick a random sample in the dataset and visualize it
    idx = np.random.randint(0, len(dataset))
    random_sample = dataset.__getitem__(idx)

    # Move sample to correct device, remove batch dimension and move to cpu for plotly
    transform_sample(random_sample, lambda x: x.clone().to(model.device))
    result = model(random_sample).cpu()
    transform_sample(random_sample, lambda x: x.cpu())

    # Order data in grids
    grid_data = get_grid(random_sample['density'].detach().numpy())
    grid_result = get_grid(result.detach().numpy())
    grid_target = get_grid(random_sample['temporal_density_gradient'].detach().numpy())

    # https://stackoverflow.com/a/58853985
    fig = make_subplots(rows=1,
                        cols=3,
                        subplot_titles=subplot_titles)
    if same_color_axis:
        fig.add_trace(go.Heatmap(z=grid_data, coloraxis = "coloraxis"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=grid_target, coloraxis = "coloraxis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=grid_result, coloraxis = "coloraxis"), row=1, col=3)
        fig.update_layout(coloraxis = {'colorscale':'plasma'})
    else:
        fig.add_trace(go.Heatmap(z=grid_data), row=1, col=1)
        fig.add_trace(go.Heatmap(z=grid_target), row=1, col=2)
        fig.add_trace(go.Heatmap(z=grid_result), row=1, col=3)

    # Add MSE loss
    loss = mse_loss(result, random_sample['temporal_density_gradient'])

    if title is None:
        fig.update_layout(title_text=f"Random batch MSE: {loss.item():.4f}")
    else:
        fig.update_layout(title_text=f"{title} <br><sup>MSE on Random Batch: {loss.item():.4f}</sup>")

    fig.update_layout(width=width, height=height)

    return fig



def visualize_model(model,
                    dataset,
                    same_color_axis = False,
                    title = None,
                    subplot_titles = ('Densities',  'Target density gradient', 'Predicted density gradient')):

    fig = visualize_model_fig(model, dataset, same_color_axis, title, subplot_titles)
    fig.show()

def fig_to_tensor(fig):
    """
    Converts a plotly or matplotlib figure to a tensor. This is useful for logging to tensorboard.
    """
    # check if figure is mathplotlib or plotly
    import io
    from PIL import Image
    from torchvision import transforms

    if isinstance(fig, matplotlib.figure.Figure):
        bytes_image = io.BytesIO()
        fig.savefig(bytes_image, format="png")
        bytes_image.seek(0)
    else:
        bytes_image = io.BytesIO(fig.to_image(format="png"))

    image = Image.open(bytes_image)
    image = transforms.ToTensor()(image)
    return image

class SaveOutputHandler:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []