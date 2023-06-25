import os
import sys
import numpy as np
import torch
import numpy as np
import zstandard as zstd
from tqdm import tqdm
import msgpack
import msgpack_numpy
from glob import glob
from utils.train_helper import generate_map, load_idx_to_file_map
msgpack_numpy.patch()


class SimulationDataset(torch.utils.data.Dataset):
    """
    Dataset for loading compressed partio files.
    """

    # See transforms and rescaling data:
    #  https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    #  Possible transforms for us: normalize, rotate, jitter, inject density
    def __init__(self, files, window=1, transform=None, transform_once=None, cache=False, device='cpu'):
        """
        Initialize the dataset from a list of compressed partio files.

        :param files: list of paths to partio files
        :param transform: transform to apply to each sample
        :param transform_once: transform to apply to each sample once, after loading it from disk
        :param cache: if True, load all data into memory
        :param device: device to load data to
        :param window: number of timesteps to return per sample (if 1, return only one timestep)
        """

        if not len(files):
            raise Exception("List of files must not be empty")
        if window < 1:
            raise Exception("window must be >=1 but is {}".format(window))
        if window > 1:
            raise Exception("window > 1 not implemented yet")
        if not all([os.path.isfile(f) for f in files]):
            raise Exception("Not all files exist")

        self.window = window
        self.transform = transform
        self.transform_once = transform_once
        self.device = device
        self.has_printed_warning = False

        # File handling
        files.sort() # this is essential. If the number of files if off things won't work
        self.files = [os.path.abspath(f) for f in files]
        self.directory = os.path.dirname(self.files[0])
        self.filemap = load_idx_to_file_map(self.directory)
        self.length = len(self.filemap)

        if len(set([f[4] for f in self.filemap])) != len(self.files):
            raise Exception("Number of files in filemap and given files do not match. Please rebuild dataset filemap.")

        # If cache is enabled, load all data into memory
        self.enable_cache = cache
        self.cache = []

        if self.enable_cache:
            print("Loading dataset into memory and applying transform_once... (Device: {})".format(self.device))
            for i in tqdm(range(len(self.files))):
                file = self.files[i]
                file_content = self._get_file_content(file)

                for i in range(len(file_content)):
                    file_content[i]['box'] = file_content[0]['box']
                    sample = self._prepare_sample(file_content[i])
                    self.cache.append(sample)

            print("Done loading dataset into memory")


    def __getitem__(self, idx):
        # A file info record tuple looks like:
        # (i, in_file_id, file_id, box_id, basename, frame_id, scene_id)

        # Load data, either cache or from file
        if self.enable_cache:
            sample = self.cache[idx]
        else:
            in_file_idx = self.filemap[idx][1]
            file_name = self.filemap[idx][4]

            data = self._get_file_content(os.path.join(self.directory, file_name))
            data[in_file_idx]['box'] = data[0]['box']
            sample = self._prepare_sample(data[in_file_idx])

        if self.transform:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.length

    def _get_file_content(self, file):
        decompressor = zstd.ZstdDecompressor()
        with open(file, 'rb') as f:
            data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
        return data

    def _prepare_sample(self, raw_sample):
        """
        Convert a raw sample to a sample that can be used by the network.
        (Tensors, correct device, correct views, etc.)

        Available keys in raw_sample:
        ['pos', 'vel', 'm', 'viscosity', 'box', 'box_normals', 'num_rigid_bodies', 'frame_id', 'scene_id']
        """

        sample = {}
        for key in ['pos', 'vel', 'm', 'viscosity', 'box']:
            sample[key] = raw_sample[key]

            if not isinstance(sample[key], torch.Tensor):
                sample[key] = torch.tensor(sample[key], dtype=torch.float32, device=self.device)

            # if last dimension is not 1 or 3, add it
            if sample[key].shape[-1] != 1 and sample[key].shape[-1] != 3:
                sample[key] = sample[key].view(-1, 1)


        if (sample['m'] == 0).all():
            if not self.has_printed_warning:
                print("WARNING: All masses are zero. Setting masses to 0.125. (This message is only shown once.)")
                self.has_printed_warning = True

            sample['m'] = torch.ones_like(sample['m']) * 0.125

        if self.transform_once:
            sample = self.transform_once(sample)

        return sample
