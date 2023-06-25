import argparse
import os.path
from utils.train_helper import generate_map

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