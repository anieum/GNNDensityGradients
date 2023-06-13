import torch
import pyvista as pv

class VtkDataset(torch.utils.data.Dataset):
    # Todo, we need the torch tensors
    # Todo, shuffle, batching, etc.
    # Todo, add velocity as data

    def __init__(self, path, jitter=0.0):
        if jitter != 0.0:
            self.add_noise(jitter)

        # todo: compose dataset from multiple files
        if not isinstance(paths, list):
            paths = [paths]
        # How do I handle multiple files? The neighbor search is done on a per-file basis, so I can't just concatenate the files.
        # I can calculate from particle counts how many particles are in each file, and then use that to index into the correct file.

        points_per_file = []
        self.n_points = 0

        for path in paths:
            mesh = pv.UnstructuredGrid(path)
            points_per_file.append(torch.tensor(mesh.n_points))
            file_idx = len(points_per_file) - 1

            # concatenate features
            densities = torch.tensor(mesh.point_data["density"])
            velocities = torch.tensor(mesh.point_data["velocity"])

            for density, velocity in zip(densities, velocities):
                sample = (density, velocity, file_idx)
                self.data.append(sample)

            # todo, continue here

            density_gradients = torch.tensor(mesh.point_data["density_grad"])
            
            self.data = data
            self.target = density_gradients

            self.n_points += mesh.n_points

        self.data.contiguous()
        self.target.contiguous()

    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        # Care: Does this return dangling references? Probably!
        return torch.tensor(self.data[idx]).view(-1), torch.tensor(self.target[idx]).view(-1)
    
    def add_noise(self, noise):
        raise Exception("TODO: Implement noise")
    
    def get_points(self):
        return torch.tensor(self.mesh.points)
    
    def normalize(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / (std + 1e-8)
    