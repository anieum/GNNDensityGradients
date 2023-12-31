import torch
import numpy as np
from torch_geometric.nn import radius_graph
from torch_geometric.nn import radius as radius_search
from torch_scatter import scatter
from functorch import vmap # care in pytorch 2.0 this will be torch.vmap
from warnings import warn

# Todo: use pysph or torchSPHv2
# Todo: smoothing length != cutoff! make sure the kernel doesn't go to 0 too early

# todo: implement QCM6-kernel from Rosswog2015 or better alternative
# todo: test kernels + derivative if correct
def apply_cubic_kernel_torch(r,h):
    """
    Implementation of cubic-b-spline kernel function.
    :param r: distance between two points. Shape: (batch, distances)
    :param h: radius for the kernel function
    :return: kernel value

    :note: You must ensure that r is positive
    """

    # 3/2pi*h^3; the 3/2 is already included in the return values
    dimensional_factor = 1 / (np.pi * h * h * h)

    zero_tensor = torch.zeros_like(r)
    result = torch.where((r >= 2.0) | (r > h) | (r < 0), zero_tensor, r)
    result = torch.where((r >= 0.0) & (r < 1.0), dimensional_factor * (1 - 1.5 * r ** 2 + 0.75 * r ** 3), result)
    result = torch.where((r >= 1.0) & (r < 2.0), dimensional_factor * 0.25 * (2 - r) ** 3, result)

    return result

# see https://www.gpusph.org/documentation/gpusph-theory.pdf eq. (13)
def apply_cubic_kernel_derivative_torch(r, h):
    """
    Implementation of the derivative of the cubic-b-spline kernel function.
    :param r: distance between two points. Shape: (batch, distances)
    :param h: radius for the kernel function
    :return: kernel value

    :note: You must ensure that r is positive
    """
    # todo: enforce r is positive via nan return value or similar

    # 3/2pi*h^3; the 3/2 is already included in the return values
    dimensional_factor = 1 / (np.pi * h * h * h)

    zero_tensor = torch.zeros_like(r)
    result = torch.where((r >= 2.0) | (r > h) | (r < 0), zero_tensor, r)
    result = torch.where((r >= 0.0) & (r < 1.0), dimensional_factor * (-3 * r + 2.25 * r * r), result)
    result = torch.where((r >= 1.0) & (r < 2.0), dimensional_factor * (-0.75 * (2 - r) ** 2), result)

    return result


def set_kernel(kernel_func):
    """
    Sets the kernel function to use.
    :param kernel: kernel function
    """
    global kernel
    kernel = kernel_func

def set_kernel_grad(kernel_grad_func):
    """
    Sets the kernel gradient function to use.
    :param kernel_grad: kernel gradient function
    """
    global kernel_grad
    kernel_grad = kernel_grad_func

def set_smoothing_length(h):
    """
    Sets the smoothing length to use.
    :param smoothing_length: smoothing length
    """
    global smoothing_length
    smoothing_length = h

set_kernel(apply_cubic_kernel_torch)
set_kernel_grad(apply_cubic_kernel_derivative_torch)
set_smoothing_length(0.05) # 0.04984372318264361

def get_neighbors(point_cloud, point_cloud_with_boundary, radius, loop=True):
    """
    Calculates the neighbors for each point in the point cloud.
    :param point_cloud: point cloud of sph particles
    :param radius: radius for the kernel function
    :return: neighbors for each point in the point cloud
    """
    # TODO ENSURE THIS OUTPUTS EXACTLY THE SAME AS
    # j, i = radius_graph(point_cloud, r=smoothing_length * 3.3, batch=None, loop=True, max_num_neighbors=2048)
    # (with j and i swapped as output here)
    # given the correct parameters

    # TODO: Support Batchdimensions via vmap! https://pytorch.org/functorch/stable/generated/functorch.vmap.html
    # it however seems to require a new pytorch version: https://github.com/pytorch/pytorch/issues/97425

    # if point_cloud.dim == 2:
    #     point_cloud = point_cloud.unsqueeze(0)

    # how to support batchdimensions? The lengths of the tensors can be different for each item in the batch
    # so we can't stack our tensors. As batch[0] could be [10, 2] and batch[1] could be [12, 2] etc.
    # TODO FIX: WE USE THE VMAP FOR GET_DENSITY, GET_DENSITY_GRADIENT etc. ITS OUTPUT DIMENSIONS ARE CONSTANT!

    pairs = radius_search(x=point_cloud_with_boundary, y=point_cloud, r=radius, max_num_neighbors=2048)

    if not loop:
        pairs = pairs[:,pairs[0] != pairs[1]]
        pairs.contiguous()  # check if this improves or worsens performance

    return pairs


def const_len_append(const, tensor_first, tensor_second):
    """
    Concats tensor_first with constant length of tensor_seconds values.
    :param const: constant value
    :param tensor_first: tensor to prepand
    :param tensor_second: tensor to get the shape from
    :return: tensor of same shape as tensor, but with all values set to const
    """

    new_shape = list(tensor_first.shape)
    new_shape[0] = tensor_second.shape[0]

    tensor_second_new = const * torch.ones(new_shape, dtype=tensor_second.dtype, device=tensor_second.device)

    return torch.concat([tensor_first, tensor_second_new])


def get_density(point_cloud, masses, kernel=kernel, smoothing_length=smoothing_length, boundary_box = None):
    """
    Calculates the density for each point in the point cloud.
    formula: rho_i = sum_j m_j W(|r_i - r_j|, h)

    :param point_cloud: point cloud of sph particles
    :param masses: masses of each point in the point cloud
    :param w: kernel function
    :param smoothing_length: smoothing length for the kernel function
    :return: density rho for each point in the point cloud
    """
    with torch.no_grad():
        total_particles = point_cloud if boundary_box is None else torch.concat([point_cloud, boundary_box])
        masses = masses if boundary_box is None else const_len_append(masses.view(-1)[0], masses, boundary_box)

        i, j = get_neighbors(
            point_cloud = point_cloud,
            point_cloud_with_boundary = total_particles,
            radius = smoothing_length * 3.3,
            loop = True,
        )

        # Calculate pairwise distances, by looking up the points in the point cloud
        distances = torch.norm(total_particles[i] - total_particles[j], dim=-1, keepdim=True)

        # Calculate pairwise kernel values
        kernel_values = kernel(distances / smoothing_length, smoothing_length)
        kernel_values = masses[j] * kernel_values

        return scatter(kernel_values, i, dim=0, dim_size = point_cloud.shape[0], reduce='add')




def get_spatial_density_gradient(point_cloud, masses, densities, kernel_grad=kernel_grad, smoothing_length=smoothing_length, boundary_box = None):
    """
    Calculates the spatial density gradient for each point in the point cloud.
    formula: Grad_i rho_i = 1/rho_i sum_j m_j (rho_i + rho_j) Grad_i W_i_j

    :param point_cloud: point cloud of sph particles
    :param masses: masses of each point in the point cloud
    :param densities: densities of each point in the point cloud
    :param kernel_grad: kernel gradient function
    :param smoothing_length: smoothing length for the kernel function
    :return: spatial density gradient Grad rho for each point in the point cloud
    """
    # TODO: check if pairs occur once or twice! (Assumption is twice.)
    total_particles = point_cloud if boundary_box is None else torch.concat([point_cloud, boundary_box])
    masses = masses if boundary_box is None else const_len_append(masses.view(-1)[0], masses, boundary_box)
    densities = densities if boundary_box is None else const_len_append(densities.view(-1)[0], densities, boundary_box)
    # TODO: Which density for boundary particles?

    i, j = get_neighbors(
        point_cloud = point_cloud,
        point_cloud_with_boundary = total_particles,
        radius = smoothing_length * 3.3,
        loop = False,
    )

    pairwise_difference = total_particles[j] - total_particles[i]
    pairwise_distances = torch.norm(pairwise_difference, dim=-1, keepdim=True)
    pairwise_directions = pairwise_difference / pairwise_distances
    pairwise_directions.nan_to_num_(0.)

    pairwise_density_diffs = densities[j] - densities[i]
    kernel_grad = kernel_grad(pairwise_distances / smoothing_length, smoothing_length) * pairwise_directions
    weight = masses[j] * pairwise_density_diffs * kernel_grad
    summed = scatter(weight, i, dim=0, dim_size = point_cloud.shape[0], reduce='add')

    return summed / densities[:len(point_cloud)]



def get_temporal_density_gradient(point_cloud, masses, velocities, kernel_grad=kernel_grad, smoothing_length=smoothing_length, boundary_box = None):
    """
    Calculates the temporal density gradient for each point in the point cloud.
    formula: Drho_i / Dt = sum_j m_j (v_i - v_j) W_Grad_i_j

    :param point_cloud: point cloud of sph particles
    :param densities: densities of each point in the point cloud
    :param volumes: volumes of each point in the point cloud
    :param velocities: velocities of each point in the point cloud
    :param smoothing_length: smoothing length for the kernel function
    :return: temporal density gradient Drho/Dt for each point in the point cloud
    """
    # ignore loops, because if i == j, then v_i - v_j = 0 => Drho_i / Dt = 0
    total_particles = point_cloud if boundary_box is None else torch.concat([point_cloud, boundary_box])
    masses = masses if boundary_box is None else const_len_append(masses.view(-1)[0], masses, boundary_box)
    velocities = velocities if boundary_box is None else const_len_append(0., velocities, boundary_box)

    i, j = get_neighbors(
        point_cloud = point_cloud,
        point_cloud_with_boundary = total_particles,
        radius = smoothing_length * 3.3,
        loop = False,
    )

    pairwise_difference = total_particles[j] - total_particles[i]
    pairwise_distances = torch.norm(pairwise_difference, dim=-1, keepdim=True)
    direction = pairwise_difference / pairwise_distances

    # Todo: Some datasets seem to contain 1 or 2 overlapping points
    # nan_idx = torch.isnan(direction).any(dim=-1)
    # if nan_idx.any():
    #     warn("Dataset contains overlapping points")
    #     # make sure to look at the unique points. i[nan_idx] is not unique
    #     print("Overlapping points: ", i[nan_idx], point_cloud[i[nan_idx]])
    direction.nan_to_num_(0.)

    kernel_grad = kernel_grad(pairwise_distances / smoothing_length, smoothing_length)
    kernel_grad = kernel_grad * direction
    velocity_diffs = velocities[i] - velocities[j]

    # multiply s.t. the resulting matrix is of shape (n, 1)
    density_diffs = torch.sum(velocity_diffs * kernel_grad, dim=-1, keepdim=True)
    density_diffs = masses[j] * density_diffs

    temporal_gradient = scatter(density_diffs, i, dim=0, dim_size = point_cloud.shape[0], reduce='add')

    return temporal_gradient



def approximate_temporal_gradient(point_cloud_1, point_cloud_2, masses, delta_t):
    """
    Approximates the temporal gradient by calculating the
    gradient change over two point clouds.
    :param point_cloud_1: point cloud of sph particles at time t
    :param point_cloud_2: point cloud of sph particles at time t + delta_t
    :param masses: masses of each point in the point cloud
    :param delta_t: time difference between the two point clouds
    :return: temporal density gradient Drho/Dt for each point in the point cloud
    """

    # todo: do this by letting the particles move and then calculating the density gradient

    density_2 = get_density(point_cloud_2, masses, w)
    density_1 = get_density(point_cloud_1, masses, w)
    return (density_2 - density_1) / delta_t



def approximate_spatial_gradient(point_cloud, masses, w, iterations, delta_x = 0.05):
    """
    Approximates the spatial gradient over multiple iterations, by
    nudging each particle in a random direction and calculating the
    average density gradient from that.
    :param point_cloud: point cloud of sph particles
    :param masses: masses of each point in the point cloud
    :param w: kernel function
    :param iterations: number of iterations
    :param delta_x: amount of nudging
    :return: spatial density gradient Drho/Dx for each point in the point cloud
    """

    # monte carlo inspired
    density_gradient = torch.zeros_like(point_cloud)

    for _ in range(iterations):
        diff = torch.random_like(point_cloud) * delta_x

        density_nudged = get_density(point_cloud + diff, masses, w)
        density = get_density(point_cloud, masses, w)

        density_gradient += (density_nudged - density).view(-1, 1) / delta_x

    return density_gradient / iterations