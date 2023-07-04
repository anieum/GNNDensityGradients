from pytorch_lightning.tuner import Tuner
from utils.visualization import fig_to_tensor
from glob import glob
import os.path
import pickle
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def find_learning_rate(trainer, model, datamodule):
    lr_finder = Tuner(trainer).lr_find(model=model, datamodule=datamodule)

    fig = fig_to_tensor(lr_finder.plot(suggest=True))
    tensorboard = trainer.logger.experiment
    tensorboard.add_image(f"results/lr_finder", fig, global_step=trainer.global_step)

    return lr_finder.suggestion()


def get_file_info(filepath):
    """
    Given a .zst file, gets all stored simulations states and returns for each (filename, frame_id, scene_id) as a list.
    """

    decompressor = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as f:
        content = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)

    basename = os.path.basename(filepath)

    # id + box id
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
    files.sort()

    file_info = []

    i = 0
    box_id = 0
    for file_id, file in enumerate(files):
        box_id = i
        in_file_id = 0
        for (basename, frame_id, scene_id) in get_file_info(file):
            # i is the number of the record in the whole dataset
            # in_file_id is the number of the record in the current file
            # file_id is the number of the file in the whole dataset
            # box_id is the number of the record that stores the box particles and refers to an i value
            file_info.append((i, in_file_id, file_id, box_id, basename, frame_id, scene_id))

            in_file_id += 1
            i += 1

    print("Found {} files with {} simulation states.".format(len(files), len(file_info)))

    if len(files) == 0:
        print("Make sure that the input directory contains *.zst files (e.g. dpi_dam_break/train NOT dpi_dam_break)")
        return

    output_path = os.path.join(filepath, '_simulation_states.pkl')

    with open(output_path, 'wb') as f:
        print("Writing map to {}".format(output_path))
        pickle.dump(file_info, f)

    return

def transform_msgpack_file(filepath, transform):
    """
    Given a .zst file, applies the given transform to all stored simulation states and writes the result back to the file.
    """
    # TODO: Implement this to allow storing of preprocessed data
    # See https://github.com/isl-org/DeepLagrangianFluids/blob/d651c6fdf2aca3fac9abe3693b20981b191b4769/datasets/create_physics_records.py#L100

    decompressor = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as f:
        content = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)

    content = transform(content)

    compressor = zstd.ZstdCompressor()
    with open(filepath, 'wb') as f:
        f.write(compressor.compress(msgpack.packb(content)))

    return


def load_idx_to_file_map(path):
    map_path = os.path.join(path, '_simulation_states.pkl')

    if not os.path.exists(map_path):
        print("File map does not exist. Generating...")
        generate_map(path)

    with open(map_path, 'rb') as f:
        # the "map" actually just is a list. Python uses arrays internally, so we still have O(n) for lookups
        file_list = pickle.load(f)

    if len(file_list) == 0:
        raise Exception("Filemap _simulation_states.pkl is empty")

    return file_list

def validate_hparams(hparams):
    """
    Validates the hyperparameters and throws an exception if they are invalid.
    """

    if not os.path.exists(hparams['save_path']):
        os.makedirs(hparams['save_path'])

    if not os.path.exists(hparams['dataset_dir']):
        raise Exception("Dataset directory does not exist")

    if hparams['load_checkpoint']:
        if not os.path.exists(hparams['load_path']):
            raise Exception("Checkpoint path does not exist")
        hparams['load_path'] = os.path.abspath(hparams['load_path'])
    else:
        hparams['load_path'] = None

    hparams['dataset_dir'] = os.path.abspath(hparams['dataset_dir'])
    hparams['save_path'] = os.path.abspath(hparams['save_path'])

def save_checkpoint(trainer, model, save_path):
    from datetime import datetime
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(save_path, time + '_version_' + str(trainer.logger.version) + '.ckpt')
    print("Saving checkpoint to {}".format(path))
    trainer.save_checkpoint(path)