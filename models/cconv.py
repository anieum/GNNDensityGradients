import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pytorch_lightning as pl
import open3d.ml.torch as ml3d
from functorch import vmap # care in pytorch 2.0 this will be torch.vmap


class CConvModel(pl.LightningModule):
    """
    Model that uses continuous convolutions to predict the density gradient.

    :param hparams: hyperparameters for the model  (TODO)
    """
    # see https://github.com/wi-re/torchSPHv2/blob/master/Cconv/1D/Untitled.ipynb

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.learning_rate = hparams['lr']
        self.layer_channels = [32, 64, 64, 3]
        self.conv_hprams = {
            'kernel_size': [4, 4, 4],
            'activation': None,
            'align_corners': True,
            'interpolation': 'linear',
            'coordinate_mapping': 'ball_to_cube_volume_preserving',
            'normalize': False,
            'window_function': None,
            'radius_search_ignore_query_points': True,
            'other_feats_channels': 0
        }
        self.particle_radius = 0.025
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)

        self.use_window = True
        self.layers = self._setup_layers()


    def _window_poly6(r_sqr):
        return torch.clamp((1 - r_sqr)**3, 0, 1)

    def _make_cconv_layer(self, name, activation=None, **kwargs):
        window_fn = None
        if self.use_window == True:
            window_fn = self._window_poly6

        cconv_layer = ml3d.layers.ContinuousConv(
            kernel_size=self.conv_hprams['kernel_size'],
            activation=self.conv_hprams['activation'],
            align_corners=self.conv_hprams['align_corners'],
            interpolation=self.conv_hprams['interpolation'],
            coordinate_mapping=self.conv_hprams['coordinate_mapping'],
            normalize=self.conv_hprams['normalize'],
            window_function=self.conv_hprams['window_function'],
            radius_search_ignore_query_points=self.conv_hprams['radius_search_ignore_query_points'],
            **kwargs
        )

        return cconv_layer

    def _setup_layers(self):
        """
        Creates the layers of the network: 3 parallel input layers, then for each given layer_channel a dense layer and
        a cconv layer.

        With default parameters:
        (conv0_fluid, conv0_obstacle, dense0_fluid), (dense1, conv1), (dense2, conv2), (dense3, conv3)

        :return: list of lists of layers, where layer[i] is a list of layers for the i-th layer of the network
        """

        # Input layer of networks (3 parallel layers) -----------------------------------------------
        layers = [[] for _ in range(len(self.layer_channels))]

        # Convolutional layer to handle fluids channels (1, vel_0, vel_1, vel_2, density, other_feats)
        conv0_fluid = self._make_cconv_layer(
                name="conv0_fluid",
                in_channels=4 + self.conv_hprams['other_feats_channels'],
                filters=self.layer_channels[0],
                activation=None
            )

        # Convolutional layer to handle obstacles channels, maybe (pos_0, pos_1, pos_2)?
        conv0_obstacle = self._make_cconv_layer(
                name="conv0_obstacle",
                in_channels=3,
                filters=self.layer_channels[0],
                activation=None
            )

        # Dense layer to handle fluids channels (1, vel_0, vel_1, vel_2, density, other_feats)
        dense0_fluid = torch.nn.Linear(
                in_features=4 + self.conv_hprams['other_feats_channels'],
                out_features=self.layer_channels[0]
            )
        torch.nn.init.xavier_uniform_(dense0_fluid.weight)
        torch.nn.init.zeros_(dense0_fluid.bias)

        layers[0].append(conv0_fluid)
        layers[0].append(conv0_obstacle)
        layers[0].append(dense0_fluid)

        # Intermediate layers of networks -----------------------------------------------------------
        actual_layer_channels = self.layer_channels
        actual_layer_channels[0] *= 3

        # iterate over subsequent pairs of channel_feature numbers
        for layer, (in_ch, out_ch) in enumerate(zip(actual_layer_channels, actual_layer_channels[1:]), start=1):
            dense_layer = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense_layer.weight)
            torch.nn.init.zeros_(dense_layer.bias)

            conv_layer = self._make_cconv_layer(
                name="conv{}".format(layer),
                in_channels=in_ch,
                filters=out_ch,
                activation=None
            )

            layers[layer].append(dense_layer)
            layers[layer].append(conv_layer)

        return layers

    def forward(self, sample):
        """
        :param sample: dict with keys 'pos', 'vel', 'density', 'temporal_density_gradient'
        :return: predicted temporal density gradient
        """
        pos = sample['pos']

        # (conv0_fluid, conv0_obstacle, dense0_fluid), (dense1, conv1), (dense2, conv2), (dense3, conv3)
        conv_fluid = self.layers[0][0]
        conv_obstacle = self.layers[0][1]
        dense_fluid = self.layers[0][2]

        # Get features in correct shape
        # fluid_features (1, vel_0, vel_1, vel_2, density, other_feats)
        # box features (pos_0, pos_1, pos_2)
        fluid_features = torch.cat(
                (torch.ones_like(sample['density']), sample['vel'], sample['density']), dim=-1
            ).view(-1, 4 + self.conv_hprams['other_feats_channels'])

        obstacle_features = sample['pos'].view(-1, 3) # TODO: THIS IS WRONG, WHAT DOES THE PAPER USE AS FEATURES?

        # Calculate first layer activations
        x1 = conv_fluid(inp_features=fluid_features, inp_positions=pos, out_positions=pos, extents=self.filter_extent)
        x2 = conv_obstacle(inp_features=obstacle_features, inp_positions=pos, out_positions=pos, extents=self.filter_extent)
        x3 = dense_fluid(fluid_features)

        x = torch.cat((x1, x2, x3), dim=-1)

        x = sample
        for layer in self.layers[1:]:
            x = self._forward_layer(x, layer)


        return x

    def _forward_old(self, x):
        # features are density in the first column followed by the velocity in the next 3 columns
        features_neighbors = torch.cat((x['density'], x['vel']), dim=-1).view(-1, 4)
        neighbors = x['pos'].view(-1, 3)  # TODO: THIS IGNORES BOUNDARIES
        out_pos = x['pos'].view(-1, 3)  # Bug for wrong shape, probably in visualization hook

        x = self.lin_embedding(features_neighbors).float()
        x = self.cconv(inp_features=x, inp_positions=neighbors, out_positions=out_pos, extents=0.6)
        x = self.lin_out_mapping(x)

        return x


    def configure_optimizers(self):
        # todo: configure scheduler
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = ReduceLROnPlateau(optimizer, ...)
        #return [optimizer], [scheduler]
        return optimizer


    def training_step(self, train_batch, batch_idx):
        # train_batch is a list of batches
        if not isinstance(train_batch, list):
            raise Exception("train_batch must be a list of batches")

        y_pred = [self(sample) for sample in train_batch]
        y_target = [sample['temporal_density_gradient'] for sample in train_batch]

        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(y_pred)):
            loss += F.mse_loss(y_pred[i], y_target[i])
        loss /= len(y_pred)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.hparams['batch_size'])
        return loss


    def validation_step(self, val_batch, batch_idx):
        y_pred = [self(sample) for sample in val_batch]
        y_target = [sample['temporal_density_gradient'] for sample in val_batch]

        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(y_pred)):
            loss += F.mse_loss(y_pred[i], y_target[i])
        loss /= len(y_pred)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.hparams['batch_size'])



