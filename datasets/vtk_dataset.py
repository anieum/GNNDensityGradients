import torch
import pyvista as pv

class VtkDataset(torch.utils.data.Dataset):
    # Todo, we need the torch tensors
    # Todo, shuffle, batching, etc.
    # Todo, add velocity as data

    def __init__(self, paths, jitter=0.0, normalize=True):
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

        self.points_per_file = []
        self.n_points = 0
        self.data = []
        self.target = []
        self.sizes = []

        for path in paths:
            mesh = pv.UnstructuredGrid(path)

            # concatenate features
            if normalize:
                densities = self.normalize(torch.tensor(mesh.point_data["density"]))
                velocities = self.normalize(torch.tensor(mesh.point_data["velocity"]))
            else:
                densities = torch.tensor(mesh.point_data["density"])
                velocities = torch.tensor(mesh.point_data["velocity"])

            features = torch.cat((densities.view(-1, 1), velocities), dim=-1)

            points = torch.tensor(mesh.points)
            self.data.append(features)
            self.points_per_file.append(points)
            self.sizes.append(len(points))

            # for density, velocity, point in zip(densities, velocities, points):
            #     sample = (density, velocity, point, file_idx)
            #     self.data.append(sample)
            
            density_gradients = torch.tensor(mesh.point_data["density_grad"])
            self.target.append(density_gradients)

            self.n_points += mesh.n_points

        # self.target = torch.concat(self.target)
        # self.target.contiguous()

        # todo: check if this works
        # self.data = torch.stack(self.data)
        # self.data.contiguous()

    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        # Care: Does this return dangling references? Probably!

        file_idx = 0
        for i, size in enumerate(self.sizes):
            if idx >= size:
                idx -= size
            else:
                file_idx = i
                break

        position = self.points_per_file[file_idx][idx]
        neighbors = self.points_per_file[file_idx]
        features_neighbors = self.data[file_idx]

        return (features_neighbors, neighbors, position), self.target[file_idx][idx]
        # return (data, density, point, self.points_per_file[file_idx]), self.target[idx]
    
    def add_noise(self, noise):
        raise Exception("TODO: Implement noise")
    
    def normalize(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / (std + 1e-8)