import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import open3d.ml.torch as ml3d
from functorch import vmap # care in pytorch 2.0 this will be torch.vmap


class CConvModel(pl.LightningModule):
    """
    Model that uses continuous convolutions to predict the density gradient.

    :param hparams: hyperparameters for the model  (TODO)
    """
    # see https://github.com/wi-re/torchSPHv2/blob/master/Cconv/1D/Untitled.ipynb

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        self.lin_embedding = nn.Linear(4,8)
        self.cconv = ml3d.layers.ContinuousConv(
		    in_channels=8,
            filters=16,
	        kernel_size=[3,3,3],
            activation=nn.ReLU(),
            align_corners=True,
            normalize=False,
            radius_search_ignore_query_points=True
	    )
        self.lin_out_mapping = nn.Linear(16, 1)


    def forward(self, x):
        # features are density in the first column followed by the velocity in the next 3 columns
        all_features_neighbors = torch.cat((x['density'], x['vel']), dim=-1)
        all_neighbors = x['pos']  # TODO: THIS IGNORES BOUNDARIES
        all_out_pos = x['pos']

        # unfortunately, this does not work. Maybe because cconv already uses the gpu internally
        # self.cconv_parallel = vmap(self.cconv, in_dims=(0, 0, 0, None))
        # x = self.cconv_parallel(x, neighbors, out_pos, 2.0)

        results = []
        if all_out_pos.dim() > 2:
            for features_neighbors, neighbors, out_pos in zip(all_features_neighbors, all_neighbors, all_out_pos):
                x = self.lin_embedding(features_neighbors).float()
                # print("SHAPES: ", x.shape, neighbors.shape, out_pos.shape)
                x = self.cconv(inp_features=x, inp_positions=neighbors, out_positions=out_pos, extents=2.0)
                x = self.lin_out_mapping(x)
                results.append(x)
        else:
            raise Exception("Not implemented")

        return torch.stack(results)


    def configure_optimizers(self):
        # todo: configure scheduler
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = ReduceLROnPlateau(optimizer, ...)
        #return [optimizer], [scheduler]
        return optimizer


    def training_step(self, train_batch, batch_idx):
        y_pred = self(train_batch)
        loss = F.mse_loss(y_pred, train_batch['temporal_density_gradient'])

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, val_batch, batch_idx):
        y_pred = self(val_batch)
        loss = F.mse_loss(y_pred, val_batch['temporal_density_gradient'])

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)



