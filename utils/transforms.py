from typing import Any
import torch
import torchvision.transforms as tf
import utils.sph
from utils.sph import *
import pytorch_lightning as pl

class SampleToTensor(pl.LightningDataModule):
    """
    Converts a sample to a tensor.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, sample):
        # torch.as_tensor() would be nicer, but the given numpy arrays are read-only, so we have to copy them
        sample['pos'] = torch.tensor(sample['pos'], dtype=torch.float32, device=self.device)
        sample['vel'] = torch.tensor(sample['vel'], dtype=torch.float32, device=self.device)
        sample['viscosity'] = torch.tensor(sample['viscosity'], dtype=torch.float32, device=self.device)
        sample['m'] = torch.tensor(sample['m'], dtype=torch.float32, device=self.device)
        sample['box'] = torch.tensor(sample['box'], dtype=torch.int32, device=self.device)
        sample['box_normals'] = torch.tensor(sample['box_normals'], dtype=torch.int32, device=self.device)

        return sample


class AddDensity(object):
    """
    Adds the density to the sample.
    """

    def __init__(self, kernel=apply_cubic_kernel_torch, smoothing_length=smoothing_length, include_box=True, device = 'cpu'):
        self.kernel = kernel
        self.smoothing_length = smoothing_length
        self.device = device
        self.include_box = include_box

    def __call__(self, sample):
        boundary_box = sample['box']

        if not self.include_box:
            boundary_box = None

        sample['density'] = get_density(
            point_cloud = sample['pos'],
            masses = sample['m'],
            kernel=self.kernel,
            smoothing_length=self.smoothing_length,
            boundary_box = boundary_box
        )

        return sample


class AddSpatialDensityGradient(object):
    """
    Adds the spatial density gradient to the sample.
    """

    def __init__(self, kernel_grad=apply_cubic_kernel_derivative_torch, smoothing_length=smoothing_length, include_box=True):
        self.kernel = kernel
        self.kernel_grad = kernel_grad
        self.smoothing_length = smoothing_length
        self.include_box = include_box


    def __call__(self, sample):
        if not 'density' in sample:
            raise Exception("Density has to be added before calculating the density gradient")

        boundary_box = sample['box']

        if not self.include_box:
            boundary_box = None

        sample['spatial_density_gradient'] = get_spatial_density_gradient(
            point_cloud = sample['pos'],
            masses = sample['m'],
            densities = sample['density'],
            kernel_grad=self.kernel_grad,
            smoothing_length=self.smoothing_length,
            boundary_box = boundary_box
        )

        return sample


class AddTemporalDensityGradient(object):
    """
    Adds the temporal density gradient to the sample.
    """

    def __init__(self, kernel_grad=apply_cubic_kernel_derivative_torch, smoothing_length=smoothing_length, include_box=True):
        self.kernel_grad = kernel_grad
        self.smoothing_length = smoothing_length
        self.include_box = include_box


    def __call__(self, sample):
        if not 'density' in sample:
            raise Exception("Density has to be added before calculating the density gradient")

        boundary_box = sample['box']

        if not self.include_box:
            boundary_box = None

        sample['temporal_density_gradient'] = get_temporal_density_gradient(
            point_cloud = sample['pos'],
            masses = sample['m'],
            velocities = sample['vel'],
            kernel_grad = self.kernel_grad,
            smoothing_length = self.smoothing_length,
            boundary_box = boundary_box
        )

        return sample