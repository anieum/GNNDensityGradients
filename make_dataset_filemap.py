import argparse
import os.path
import pickle
from glob import glob
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def get_file_info(filepath):
    """
    Given a .zst file, gets all stored simulations states and returns for each (filename, frame_id, scene_id) as a list.
    """

    decompressor = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as f:
        content = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)

    basename = os.path.basename(filepath)

    return [(basename, simulation['frame_id'], simulation['scene_id']) for simulation in content]


def generate_map(filepath):
    """
    Generates a list of all simulation states in the given directory extracted from all *.zst files.
    """
    if not os.path.exists(filepath):
        raise Exception("Data directory does not exist")

    if os.path.isfile(filepath):
        raise Exception("Input path must be a directory")

    files = glob(os.path.join(filepath, '*.zst'))

    file_info = []
    for file in files:
        file_info += get_file_info(file)

    print("Found {} files with {} simulation states.".format(len(files), len(file_info)))

    if len(files) == 0:
        print("Make sure that the input directory contains *.zst files (e.g. dpi_dam_break/train NOT dpi_dam_break)")
        return

    output_path = os.path.join(filepath, '_simulation_states.pkl')

    with open(output_path, 'wb') as f:
        print("Writing map to {}".format(output_path))
        pickle.dump(file_info, f)

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract simulation states from compressed partio files.\n\n"
        "E.g.: python make_dataset_filemap.py 'datasets/data/dpi_dam_break/train'")
    parser.add_argument("input_path", help="Path to folder with *.zst files", )
    args = parser.parse_args()

    # check if folder exists
    if not os.path.exists(args.input_path):
        print("Input folder does not exist: {}".format(args.input_path))
        exit(1)

    path = os.path.abspath(args.input_path)
    generate_map(path)
    exit(0)