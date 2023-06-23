import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import open3d.ml.torch as ml3d



class CConvModel(pl.LightningModule):
    """
    Model that uses continuous convolutions to predict the density gradient.

    :param hparams: hyperparameters for the model  (TODO)
    """
    # see https://github.com/wi-re/torchSPHv2/blob/master/Cconv/1D/Untitled.ipynb

    def __init__(self):
        super().__init__()

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
        features_neighbors = torch.cat((x['density'], x['vel']), dim=-1)
        neighbors = x['pos']  # TODO: THIS IGNORES BOUNDARIES
        out_pos = x['pos']

        x = self.lin_embedding(features_neighbors).float()
        x = self.cconv(inp_features=x, inp_positions=neighbors, out_positions=out_pos, extents=2.0)
        x = self.lin_out_mapping(x)

        return x


    def configure_optimizers(self):
        # todo: configure scheduler
        optimizer = Adam(self.parameters(), lr=1e-3)
        #scheduler = ReduceLROnPlateau(optimizer, ...)
        #return [optimizer], [scheduler]
        return optimizer


    def training_step(self, train_batch, batch_idx):
        y_pred = self(train_batch)
        loss = F.mse_loss(y_pred, train_batch['temporal_density_grad'].unsqueeze(-1))

        return loss


    def validation_step(self, val_batch, batch_idx):
        y_pred = self(val_batch)
        loss = F.mse_loss(y_pred, val_batch['temporal_density_grad'].unsqueeze(-1))

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)



