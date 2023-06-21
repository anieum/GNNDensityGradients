import torch
import numpy as np
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

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
set_smoothing_length(0.9)

def get_density(point_cloud, masses, kernel=kernel, smoothing_length=smoothing_length):
    """
    Calculates the density for each point in the point cloud.
    formula: rho_i = sum_j m_j W(|r_i - r_j|, h)

    :param point_cloud: point cloud of sph particles
    :param masses: masses of each point in the point cloud
    :param w: kernel function
    :param smoothing_length: smoothing length for the kernel function
    :return: density rho for each point in the point cloud
    """

    j, i = radius_graph(point_cloud, r=smoothing_length, batch=None, loop=True, max_num_neighbors=256)

    # Calculate pairwise distances, by looking up the points in the point cloud
    distances = torch.norm(point_cloud[i] - point_cloud[j], dim=1).view(-1, 1)

    # Calculate pairwise kernel values
    kernel_values = kernel(distances, smoothing_length)
    kernel_values = masses[j].view(-1, 1) * kernel_values

    return scatter(kernel_values, i, dim=0, dim_size = point_cloud.shape[0], reduce='add')



def get_spatial_density_gradient(point_cloud, masses, densities, kernel_grad=kernel_grad, smoothing_length=smoothing_length):
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
    j, i = radius_graph(point_cloud, r=smoothing_length, batch=None, loop=True, max_num_neighbors=256)

    pairwise_difference = point_cloud[j] - point_cloud[i]
    pairwise_distances = torch.norm(pairwise_difference, dim=1).view(-1, 1)
    pairwise_directions = pairwise_difference / pairwise_distances
    pairwise_directions = torch.nan_to_num(pairwise_directions, nan=1.0)

    pairwise_density_diffs = densities[j].view(-1, 1) - densities[i].view(-1, 1)
    kernel_grad = kernel_grad(pairwise_distances, smoothing_length) * pairwise_directions
    weight = masses[j].view(-1, 1) * pairwise_density_diffs * kernel_grad
    summed = scatter(weight, i, dim=0, dim_size = point_cloud.shape[0], reduce='add')

    return summed / densities.view(-1, 1)



def get_temporal_density_gradient(point_cloud, masses, velocities, kernel_grad=kernel_grad, smoothing_length=smoothing_length):
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
    j, i = radius_graph(point_cloud, r=smoothing_length, batch=None, loop=False, max_num_neighbors=256)
    
    pairwise_distances = torch.norm(point_cloud[i] - point_cloud[j], dim=1)
    kernel_grad = kernel_grad(pairwise_distances, smoothing_length)
    velocity_diffs = velocities[i] - velocities[j]

    # direction = velocity_diffs / torch.norm(velocity_diffs, dim=1, keepdim=True)
    # density_diffs = velocity_diffs * direction * kernel_grad.view(-1,1)

    density_diffs = velocity_diffs * kernel_grad.view(-1,1)
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