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