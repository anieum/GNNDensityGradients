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
from utils.transforms import ToSample
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

        If the dataset already is transformed and has density and density gradients, additional
        transformations are not necessary. If not, the following transformations are recommended:
        - AddDensity
        - AddTemporalDensityGradient
        - NormalizeDensityData
        
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
        self.to_sample_func = ToSample(device=self.device)

        # File handling
        files.sort() # this is essential. The filemap assumes that the files are sorted.
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
                    file_content[i]['box_normals'] = file_content[0]['box_normals']
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
            data[in_file_idx]['box_normals'] = data[0]['box_normals']
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
        sample = self.to_sample_func(raw_sample)

        if self.transform_once:
            sample = self.transform_once(sample)

        return sample

    def to(self, device):
        self.device = device

        if self.enable_cache:
            for i in range(len(self.cache)):
                for key in self.cache[i]:
                    if isinstance(self.cache[i][key], torch.Tensor) and self.cache[i][key].device != self.device:
                        self.cache[i][key] = self.cache[i][key].to(self.device)

        return self
