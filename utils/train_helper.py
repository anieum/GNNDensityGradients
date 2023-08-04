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
    frame_id = -1
    scene_id = -1

    return [(basename,
             simulation['frame_id'] if 'frame_id' in simulation else -1,
             simulation['scene_id'] if 'scene_id' in simulation else -1) for simulation in content]


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

    for i in range(1, len(content)):
        content[i]['box'] = content[0]['box']
        content[i]['box_normals'] = content[0]['box_normals']

    content = [transform(sample) for sample in content]

    for i in range(1, len(content)):
        content[i]['box'] = None
        content[i]['box_normals'] = None

    compressor = zstd.ZstdCompressor(level=22)
    with open(filepath, 'wb') as f:
        f.write(compressor.compress(msgpack.packb(content, use_bin_type=True)))

    return


def prepocess_dataset_files(path, type='temp_grad', include_box=False, device = 'cpu'):
    from tqdm import tqdm
    from utils.transforms import ToSample, AddDensity, AddTemporalDensityGradient, AddSpatialDensityGradient, NormalizeDensityData, ToNumpy
    from torchvision.transforms import Compose
    import torch.cuda

    if not os.path.exists(path):
        raise Exception("Data directory does not exist")

    if os.path.isfile(path):
        raise Exception("Input path must be a directory")

    if type not in ['temp_grad', 'spatial_grad', 'both']:
        raise Exception("Unknown type")

    if device != 'cuda' and torch.cuda.is_available():
        print("You can accelerate the preprocessing by using cuda.")

    files = glob(os.path.join(path, '*.zst'))
    files.sort()

    if len(files) == 0:
        raise Exception("Make sure that the input directory contains *.zst files (e.g. dpi_dam_break/train NOT dpi_dam_break)")

    # Put together transformations that are applied to each simulation state in each file
    transformations = [ToSample(device=device)]
    transformations += [AddDensity(include_box=False)]

    if type == 'temp_grad' or type == 'both':
        transformations += [AddTemporalDensityGradient(include_box=include_box)]
    if type == 'spatial_grad' or type == 'both':
        transformations += [AddSpatialDensityGradient(include_box=include_box)]

    transformations += [NormalizeDensityData()]
    transformations += [ToNumpy()]

    transformations = Compose(transformations)

    # Overwrite files
    for file in tqdm(files):
        transform_msgpack_file(file, transformations)

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

    if len(glob(os.path.join(hparams['dataset_dir'], '*.zst'))) == 0:
        raise Exception("Dataset directory does not contain *.zst files")

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

def load_hparams(file_path):
    """
    Loads the hyperparameters from a file.
    """

    import json, copy

    with open(file_path, 'r') as file:
        content = file.read()

    content = json.loads(content)

    hparams = content['lightning_config']['_module_init_config']['hparams']

    return copy.deepcopy(hparams)

def update_hparams(hparams, new_hparams):
    """
    Updates the hyperparameters with the given new parameters.
    """
    import copy

    hparams_to_ignore = ['load_checkpoint', 'save_path', 'load_path', 'params_path', 'model', 'cache', 'shuffle'
                         'dataset_dir', 'num_epochs', 'limit_train_batches', 'limit_val_batches', 'device',
                         'data_split', 'shuffle', 'batch_size', 'num_workers', 'num_training_nodes', 'num_gpus',
                         'num_training_nodes', 'num_workers', 'num_epochs', 'limit_train_batches', 'limit_val_batches'
                         'log_every_n_steps', 'val_every_n_epoch']

    for key in new_hparams.keys():
        if key in hparams_to_ignore:
            continue

        if key in hparams:
            hparams[key] = new_hparams[key]
        else:
            print("WARNING: Parameter {} does not exist".format(key))

    return copy.deepcopy(hparams)

def count_parameters(model):
    """
    Counts the number of trainable parameters in the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)