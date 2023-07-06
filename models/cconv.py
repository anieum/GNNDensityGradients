import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pytorch_lightning as pl
import open3d.ml.torch as ml3d
from functorch import vmap # care in pytorch 2.0 this will be torch.vmap
import torch.nn.functional as F


class CConvModel(pl.LightningModule):
    """
    Model that uses continuous convolutions to predict the density gradient.

    :param hparams: hyperparameters for the model  (TODO)
    """
    # see https://github.com/isl-org/DeepLagrangianFluids/blob/master/models/default_torch.py
    # see https://github.com/wi-re/torchSPHv2/blob/master/Cconv/1D/Untitled.ipynb


    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.learning_rate = hparams['lr']
        self.layer_channels = [32, 64, 64, 1]
        self.conv_hprams = {
            'kernel_size': torch.tensor([4, 4, 4], device=self.device),
            'activation': None,
            'align_corners': True,
            'interpolation': 'linear',
            'coordinate_mapping': 'ball_to_cube_volume_preserving',
            'normalize': False,
            'window_function': self._window_poly6,
            'radius_search_ignore_query_points': True,
            'other_feats_channels': 1  # 1 because we include density
        }
        self.particle_radius = 0.025
        self.radius_scale = 1.5
        self.filter_extent = float(self.radius_scale * 6 * self.particle_radius)

        self.use_window = True
        self.layers = self._setup_layers()

        # register layers as parameters, so built-in pytorch functions like .parameters() and .to() work
        self.param_list = torch.nn.ParameterList([sublayer for layer in self.layers for sublayer in layer])
        self.save_hyperparameters()

    def _window_poly6(self, r_sqr):
        return torch.clamp((1 - r_sqr)**3, 0, 1)


    def _make_cconv_layer(self, name, **kwargs):
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
        # Convolutional layer to handle fluids channels (1, vel_0, vel_1, vel_2, density, other_feats)
        conv0_fluid = self._make_cconv_layer(
                name="conv0_fluid",
                in_channels=4 + self.conv_hprams['other_feats_channels'],
                filters=self.layer_channels[0],
            )

        # Convolutional layer to handle obstacles channels, (normal_0, normal_1, normal_2) <- this is a 1 hot vector
        conv0_obstacle = self._make_cconv_layer(
                name="conv0_obstacle",
                in_channels=3,
                filters=self.layer_channels[0],
            )

        # Dense layer to handle fluids channels (1, vel_0, vel_1, vel_2, density, other_feats)
        dense0_fluid = torch.nn.Linear(
                in_features=4 + self.conv_hprams['other_feats_channels'],
                out_features=self.layer_channels[0]
            )
        torch.nn.init.xavier_uniform_(dense0_fluid.weight)
        torch.nn.init.zeros_(dense0_fluid.bias)

        layers = []
        layers.append((conv0_fluid, conv0_obstacle, dense0_fluid))

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
            )

            layers.append((dense_layer, conv_layer))

        return layers


    def forward(self, sample):
        """
        Forward pass of the network.

        :param sample: dict with keys 'pos', 'vel', 'density', 'temporal_density_gradient'
        :return: predicted temporal density gradient
        """

        # (conv0_fluid, conv0_obstacle, dense0_fluid), (dense1, conv1), (dense2, conv2), (dense3, conv3)
        conv_fluid, conv_obstacle, dense_fluid = self.layers[0]

        # Get features in correct shape
        # fluid_features (1, vel_0, vel_1, vel_2, density, other_feats)
        fluid_features = torch.cat(
                (torch.ones_like(sample['density']), sample['vel'], sample['density']), dim=-1
        ).view(-1, 4 + self.conv_hprams['other_feats_channels'])

        # Calculate first layer activations
        #pos = torch.concat((sample['pos'], sample['box']), dim=0)
        pos = sample['pos']
        x1 = conv_fluid(inp_features=fluid_features.float(), inp_positions=pos, out_positions=pos, extents=self.filter_extent)
        x2 = conv_obstacle(inp_features=sample['box_normals'], inp_positions=sample['box'], out_positions=pos, extents=self.filter_extent)
        x3 = dense_fluid(fluid_features)

        x = torch.cat((x1, x2, x3), dim=-1)

        # Calculate activations for subsequent layers
        for dense_layer, conv_layer in self.layers[1:]:
            # between layers and only between layers, apply relu
            x = F.relu(x)

            dense_x = dense_layer(x)
            conv_x = conv_layer(inp_features=x, inp_positions=pos, out_positions=pos, extents=self.filter_extent)

            # If shape unchanged, add residual connection. (This happens in intermediate layers, not e.g. in the first layer)
            x = conv_x + dense_x + x if x.shape[-1] == dense_x.shape[-1] else conv_x + dense_x

        # the original paper uses (1.0 / 128) as factor to better match the target value distribution
        # we replace this with 0.5 and get a std deviation of about 1, which matches the normalized density gradient
        return 0.5 * x


    def configure_optimizers(self):
        # todo: configure scheduler
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = ReduceLROnPlateau(optimizer, ...)
        #return [optimizer], [scheduler]
        return optimizer

    def _calculate_batch_loss(self, batch):
        y_pred = [self(sample) for sample in batch]
        y_target = [sample['temporal_density_gradient'] for sample in batch]

        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(y_pred)):
            loss += F.mse_loss(y_pred[i], y_target[i])
        loss /= len(y_pred)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._calculate_batch_loss(train_batch)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.hparams['batch_size'])
        return loss


    def validation_step(self, val_batch, batch_idx):
        loss = self._calculate_batch_loss(val_batch)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.hparams['batch_size'])


