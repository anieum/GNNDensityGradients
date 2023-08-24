from typing import Any
import torch
import utils.sph
from utils.sph import *
import pytorch_lightning as pl
from functorch import vmap # care in pytorch 2.0 this will be torch.vmap

class ToSample(object):
    """
    Convert a raw sample to a sample that can be used by the network.
    (Tensors, correct device, correct views, etc.)

    Available keys in raw_sample:
    ['pos', 'vel', 'm', 'viscosity', 'box', 'box_normals', 'num_rigid_bodies', 'frame_id', 'scene_id']
    """

    def __init__(self, device):
        self.device = device
        self.has_printed_warning = False

    def __call__(self, raw_sample):
        sample = {}
        ignore_keys = ['num_rigid_bodies', 'scene_id'] # 'frame_id',
        for key in raw_sample.keys():
            if key in ignore_keys:
                continue

            sample[key] = raw_sample[key]

            if not isinstance(sample[key], torch.Tensor) or sample[key].device != self.device:
                sample[key] = torch.tensor(sample[key], dtype=torch.float32, device=self.device)

            # if last dimension is not 1 or 3, add it
            if key != 'frame_id' and sample[key].shape[-1] != 1 and sample[key].shape[-1] != 3:
                sample[key] = sample[key].view(-1, 1)

        if (sample['m'] == 0).all():
            if not self.has_printed_warning:
                print("WARNING: All masses are zero. Setting masses to 0.125. (This message is only shown once.)")
                self.has_printed_warning = True

            sample['m'] = torch.ones_like(sample['m']) * 0.125

        return sample

class ToNumpy(object):
    """
    Convert a sample to a numpy sample.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].detach().cpu().numpy()

        return sample

class AddDensity(object):
    """
    Adds the density to the sample.
    """

    def __init__(self, kernel=apply_cubic_kernel_torch, smoothing_length=smoothing_length, include_box=True):
        self.kernel = kernel
        self.smoothing_length = smoothing_length
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

def normalize(x, dim = 0):
    mean = torch.mean(x, dim=dim)
    std = torch.std(x, dim=dim)
    return (x - mean) / (std + 1e-8)

class NormalizeDensityData(object):
    """
    Normalizes the density data.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in ['density', 'spatial_density_gradient', 'temporal_density_gradient']:
            if not key in sample:
                continue

            sample[key] = normalize(sample[key])

        return sample

def add_random_walk_noise(x, noise_std):
    noise = torch.randn_like(x)
    noise = noise * noise_std
    noise = torch.cumsum(noise, dim=0)
    return x + noise

class CorruptAttribute(object):
    """
    Corrupts the key of the sample, by adding uniform noise.
    """
    # Note: DeepMind uses accumulating random walk noise. But they also store for each particle
    # the position over 5 timesteps, so accumulating the noise there makes sense.

    def __init__(self, key, noise_std=0.01):
        self.noise_std = noise_std
        self.key = key

    def __call__(self, sample):
        sample[self.key] = sample[self.key] + torch.rand_like(sample[self.key]) * self.noise_std

        return sample