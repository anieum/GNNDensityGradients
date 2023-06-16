import argparse
import os.path
import re
import pyvista as pv
import numpy as np



def get_next_filepath(path):
    """
    Returns the path of the next vtk file in the sequence.
    Expects a naming convention of <name><number>.vtk
    """
    path = os.path.abspath(path)

    if len(os.path.splitext(path)) == 1:
        raise Exception("Make sure the input file has a .vtk extension.")

    path = os.path.splitext(path)[0]

    match = re.search(r'\d+$', path)
    int(match.group())

    return path[:-len(match.group())] + str(int(match.group()) + 1) + ".vtk"

def next_file_exists(path):
    """
    Returns true if the next file in the sequence exists.
    Expects a naming convention of <name><number>.vtk
    """
    next_file = get_next_filepath(path)

    return os.path.isfile(next_file)


def set_density_gradient(path, timestep, last_densities = None):
    """
    Adds a density gradient to the given vtk file sequence.
    Expects a naming convention of <name><number>.vtk
    """

    # Read file
    mesh = pv.UnstructuredGrid(path)
    densities = mesh.point_data["density"]

    dens_grad = np.zeros(mesh.n_points, dtype=float)

    # Calculate density gradient, if possible
    if last_densities is not None:
        dens_grad = (densities - last_densities) / timestep

    mesh.point_data.set_array(dens_grad, name="density_grad")

    # Write file
    mesh.save(path)

    # Write next file
    if next_file_exists(path):
        return set_density_gradient(get_next_filepath(path), timestep, densities) + 1

    return 1


def get_radius_no_self_loop(pos, radius):
    """
    Computes the radius of the k nearest neighbors for each particle.
    :param pos: The positions of the particles
    :param k: The number of neighbors
    :return: The radius of the k nearest neighbors for each particle
    """
    # TODO: There also is a nn.radius_graph
    adj_matrix = torch_geometric.nn.radius(pos, pos, radius, max_num_neighbors = 1000)
    no_self_loops = adj_matrix[0] != adj_matrix[1]
    return adj_matrix[0][no_self_loops], adj_matrix[1][no_self_loops]

def calculate_density(particle_container, r, h, W):
    """
    Calculates the density of the particles in the particle container.
    :param particle_container: The particle container
    :param r: The radius for the density calculation
    :param h: The smoothing length
    """
    # Formula: rho_i = sum_j m_j * W(|r_i - r_j|, h)
    
    pos = particle_container.get_positions()
    org_idx, partner_idx = get_radius_no_self_loop(pos, r)

    assert len(pos) == len(particle_container.particles), "The number of particles changed"
    assert (org_idx != partner_idx).all(), "The radius graph contains self loops"

    distances = torch.norm(pos[org_idx] - pos[partner_idx], dim=1)
    kernel_values = W(distances, h)                # TODO: How are smoothing lengths connected to the radius?
    # assert (partner_idx.shape[0] != 0) and ( kernel_values != distances).any(), "Kernel did not change the distances"

    m = particle_container.get_masses()
    weighted_kernel_values = m[partner_idx] * kernel_values    #  [:, None]

    density = torch.zeros(len(pos), dtype=pos.dtype)
    density.scatter_add_(0, org_idx, weighted_kernel_values)  # [:, 0]

    particle_container.density = density #TODO: FIX, MAKE THIS A FUNCTION
    # assert (particle_container.density != 0).all(), "Some densities are still zero"

    global densities
    densities.append(density.mean().item())

from torch_scatter import scatter
def computeDensity(particles, particleArea, particleSupport, fluidRadialDistances, fluidNeighbors):
    pairWiseDensity = particleArea[fluidNeighbors[1]] * kernel(fluidRadialDistances, particleSupport)
    fluidDensity = scatter(pairWiseDensity, fluidNeighbors[0], dim=0, dim_size = particles.shape[0],reduce='add')
    
    return fluidDensity

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Add density gradient to a given file")
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("-t", "--timestep", help="Timestep to add density gradient to", type=float, default=0.005)

    # parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    print(args.input_file, args.timestep)

    if not os.path.isfile(args.input_file):
        print("Input file does not exist.")
        return

    path = os.path.abspath(args.input_file)

    number_of_files = set_density_gradient(path, args.timestep)
    print("Added density gradient to {} files.".format(number_of_files))
    return



if __name__ == "__main__":
    main()