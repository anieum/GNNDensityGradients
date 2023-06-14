import torch
import pyvista as pv

class VtkDataset(torch.utils.data.Dataset):
    # Todo, we need the torch tensors
    # Todo, shuffle, batching, etc.
    # Todo, add velocity as data

    def __init__(self, path, jitter=0.0, normalize=False):
        """
        Initialize the dataset from a list of vtk files.
        We store densities and velocities as features and density gradients as targets.
        All data is normalized. (todo: this should be done as an extra step)

        :param paths: list of paths to vtk files
        :param jitter: amount of noise to add to the data
        :param normalize: whether to normalize the data
        """

        if jitter != 0.0:
            self.add_noise(jitter)

        # todo: compose dataset from multiple files
        if not isinstance(paths, list):
            paths = [paths]

        points_per_file = []
        self.n_points = 0
        self.data = []
        self.target = []

        for path in paths:
            mesh = pv.UnstructuredGrid(path)
            points_per_file.append(torch.tensor(mesh.n_points))
            file_idx = len(points_per_file) - 1

            # concatenate features
            if normalize:
                densities = self.normalize(torch.tensor(mesh.point_data["density"]))
                velocities = self.normalize(torch.tensor(mesh.point_data["velocity"]))
            else:
                densities = torch.tensor(mesh.point_data["density"])
                velocities = torch.tensor(mesh.point_data["velocity"])


            for density, velocity in zip(densities, velocities):
                sample = (density, velocity, file_idx)
                self.data.append(sample)
            
            # ok, mal überlegen, jetzt speichert jedes sample gleichzeitig einen file_idx. d.h. ich muss

            density_gradients = torch.tensor(mesh.point_data["density_grad"])
            self.target.append(density_gradients)

            self.n_points += mesh.n_points

        self.target = torch.stack(self.target)
        self.target.contiguous()

        # todo: check if this works
        # self.data = torch.stack(self.data)
        # self.data.contiguous()

    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        # Care: Does this return dangling references? Probably!
        data, density, file_idx = self.data[idx]

        return (data, density, self.points_per_file[file_idx]), self.target[idx]
    
    def add_noise(self, noise):
        raise Exception("TODO: Implement noise")
    
    def normalize(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / (std + 1e-8)
    