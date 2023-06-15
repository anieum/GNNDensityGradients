import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
# Result heatmap

# Build grid with all particles


def get_grid(vector):
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

def visualize_model_fig(model,
                    dataset,
                    same_color_axis = False,
                    title = None,
                    subplot_titles = ('Densities',  'Target density gradient', 'Predicted density gradient')):
    # Compare output
    features = dataset.data
    particle_positions = dataset.points_per_file

    # For each input simulation state, create output grids
    figs = []
    for features, particle_positions, targets in zip(dataset.data, dataset.points_per_file, dataset.target):
        densities = features[:, 0]
        # velocities = features[:, 1:3] # not required but useful to know

        # Forward pass
        input = ([features], [particle_positions], particle_positions)
        result = model(input)

        grid_result = get_grid(result.detach().numpy())
        grid_target = get_grid(targets)
        grid_data = get_grid(densities)


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
        
        if title is not None:
            fig.update_layout(title_text=title)

        figs.append(fig)
    
    return figs

def visualize_model(model,
                    dataset,
                    same_color_axis = False,
                    title = None,
                    subplot_titles = ('Densities',  'Target density gradient', 'Predicted density gradient')):
    
    fig = visualize_model_fig(model, dataset, same_color_axis, title, subplot_titles)
    fig.show()

def fig_to_tensor(fig):
    """
    Converts a plotly figure to a tensor. This is useful for logging to tensorboard.
    """

    import io
    from PIL import Image
    from torchvision import transforms
    bytes_image = fig.to_image(format="png")
    image = Image.open(io.BytesIO(bytes_image))
    image = transforms.ToTensor()(image)
    return image