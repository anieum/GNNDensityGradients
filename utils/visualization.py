import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from torch.nn.functional import mse_loss
import matplotlib


def plot_particles(positions, color=None, title=None, colorscale = 'Viridis', return_fig = False):
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

    if title is not None:
        fig.update_layout(title=title)

    if return_fig:
        return fig
    else:
        fig.show()

def plot_sample_for_tensorboard(model, dataset):
    random_sample, result = forward_random_sample(model, dataset)
    random_sample['temporal_predicted_gradient'] = result.reshape(-1, 1)
    # random_sample['spatial_predicted_gradient'] = torch.norm(result, dim=-1).reshape(-1, 1)
    random_sample.pop('vel')

    return plot_sample(random_sample, return_fig=True)

def plot_sample(sample, height = 400, width = 1250, return_fig = False):
    s = sample

    keys = ['vel', 'density', 'temporal_density_gradient', 'spatial_density_gradient', 'temporal_predicted_gradient', 'spatial_predicted_gradient']
    key_to_title = {
        'pos': 'Velocity',
        'density': 'Density',
        'temporal_density_gradient': 'Temporal Density Gradient',
        'spatial_density_gradient': 'Spatial Density Gradient',
        'temporal_predicted_gradient': 'Temporal Predicted Gradient',
        'spatial_predicted_gradient': 'Spatial Predicted Gradient',
    }

    number_of_keys_in_sample = sum([key in s for key in keys])

    figs = []
    if 'vel' in s:
        figs.append(plot_particles(s['pos'], torch.norm(s['vel'], dim=-1), colorscale='Viridis', return_fig=True))

    if 'density' in s:
        figs.append(plot_particles(s['pos'], s['density'], colorscale='Plasma', return_fig=True))

    if 'temporal_density_gradient' in s:
        figs.append(plot_particles(s['pos'], s['temporal_density_gradient'], colorscale='thermal', return_fig=True))

    if 'spatial_density_gradient' in s:
        figs.append(plot_particles(s['pos'], torch.norm(s['spatial_density_gradient'], dim=-1), colorscale='magma', return_fig=True))

    if 'temporal_predicted_gradient' in s:
        figs.append(plot_particles(s['pos'], s['temporal_predicted_gradient'], colorscale='thermal', return_fig=True))

    if 'spatial_predicted_gradient' in s:
        figs.append(plot_particles(s['pos'], torch.norm(s['spatial_predicted_gradient'], dim=-1), colorscale='magma', return_fig=True))

    types = [f.data[0].type for f in figs]
    specs_dict = [{'type': t} for t in types]
    titles = [key_to_title[key] for key in keys if key in s]
    fig = make_subplots(rows=1, cols=number_of_keys_in_sample, specs=[specs_dict], subplot_titles=titles)

    for i, f in enumerate(figs):
        fig.add_trace(f.data[0], row=1, col=i+1)
        fig.data[i].name = titles[i][:20]

    fig.update_layout(height=height, width=width)

    if return_fig:
        return fig
    else:
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


def forward_random_sample(model, dataset, sample_idx = None):
    """
    Returns a random sample from the dataset.
    """
    
    idx = sample_idx
    if idx is None:
        idx = np.random.randint(0, len(dataset))

    random_sample = dataset.__getitem__(idx).copy()
    print("Random sample idx:", idx)

    # Move sample to correct device, remove batch dimension and move to cpu for plotly
    transform_sample(random_sample, lambda x: x.clone().to(model.device))
    result = model(random_sample).cpu()
    transform_sample(random_sample, lambda x: x.cpu())
    return random_sample, result.detach().numpy()




def visualize_model_fig(model,
                    dataset,
                    same_color_axis = False,
                    title = None,
                    subplot_titles = ('Densities',  'Target density grad', 'Predicted density grad'),
                    width = 1000,
                    height = 450,
                    sample_idx = None):
    # Pick a random sample in the dataset and visualize it
    random_sample, result = forward_random_sample(model, dataset, sample_idx)
    if result.shape[-1] == 3:
            raise Exception("Model output has 3 channels. This is not supported.")

    # Order data in grids
    grid_data = get_grid(random_sample['density'].detach().numpy())
    grid_result = get_grid(result)
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
    loss = mse_loss(torch.tensor(result).view(-1, 1), random_sample['temporal_density_gradient'])

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