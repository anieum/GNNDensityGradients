import os
import sys
import numpy as np
import torch
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import random

# TODO: SOURCE PRANTL
class SimulationDataset(torch.utils.data.Dataset):
    """
    Dataset for loading compressed partio files.
    """


    # See transforms and rescaling data:
    #  https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    #  Possible transforms for us: normalize, rotate, jitter, inject density

    # TODO: Add source notice
    def __init__(self, files, shuffle_in_files=False, max_per_file=-1, window=1, transform=None, in_memory=False):
        """
        Initialize the dataset from a list of compressed partio files.

        :param files: list of paths to partio files
        :param shuffle_in_files: whether to shuffle the samples in each file
        :param max_per_file: maximum number of samples to return per file (if -1, return all samples)
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

        self.files = [os.path.abspath(f) for f in files]
        self.shuffle_in_files = shuffle_in_files
        self.window = window
        self.max_per_file = max_per_file
        self.transform = transform

        self.in_memory = in_memory
        self.files_mem = []

        if in_memory:
            print("Loading dataset into memory")
            for file in self.files:
                decompressor = zstd.ZstdDecompressor()
                with open(file, 'rb') as f:
                    data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
                    self.files_mem.append(data)

            print("Done loading dataset into memory")



    def __getitem__(self, idx):
        # for each idx return a file!

        data = None
        if not self.in_memory:
            decompressor = zstd.ZstdDecompressor()
            with open(self.files[idx], 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
        else:
            data = self.files_mem[idx]

        # A single file has e.g. 10 timesteps or more.
        # This is a problem, because we want to return single samples
        # but we do not know how many samples we have in each file

        # A possible fix for this is to generate a map for the dataset, that
        # translates the idx to the file and the sample in the file.

        # This however requires us to load the complete 34GB dataset and check each file
        # But this has only to be done once. We than save the map and change this function.

        # Alternatively, __getitem__ ignores the idx and implements a generator similar to Prantls.
        # E.g. yielding single samples and continuing later.

        bounding_box = data[0]['box']
        bounding_box_normals = data[0]['box_normals']

        data_idx = np.arange(len(data))

        if self.shuffle_in_files:
            np.random.shuffle(data_idx)

        if self.max_per_file > 0:
            data_idx = data_idx[:self.max_per_file]

        # ['pos', 'vel', 'm', 'viscosity', 'box', 'box_normals', 'num_rigid_bodies', 'frame_id', 'scene_id']
        # Our samples are too big. The samplelist easily takes up GBs of memory when being processed

        samples = []
        for i in data_idx:
            sample = {
                'pos': data[i]['pos'],
                'vel': data[i]['vel'],
                'm': data[i]['m'],
                'viscosity': data[i]['viscosity'],
                'box':bounding_box,
                'box_normals': bounding_box_normals,
                'frame_id': data[i]['frame_id'],
                'scene_id': data[i]['scene_id']
            }

            samples.append(sample)

        # TODO: MAKE IDX REFER A SPECIFIC SAMPLE IN A SPECIFIC FILE
        # CURRENTLY EACH IDX CAUSES A COMPLETE FILE TO BE LOADED
        if self.transform:
            return self.transform(random.choice(samples))
        else:
            return random.choice(samples)

    def __len__(self):
        return len(self.files)

    # TODO: These as transforms
    def add_noise(self, noise):
        raise Exception("TODO: Implement noise")

    def normalize(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / (std + 1e-8)