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

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.learning_rate = hparams['lr']

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
        features_neighbors = torch.cat((x['density'], x['vel']), dim=-1).view(-1, 4)
        neighbors = x['pos'].view(-1, 3)  # TODO: THIS IGNORES BOUNDARIES
        out_pos = x['pos'].view(-1, 3)  # Bug for wrong shape, probably in visualization hook

        x = self.lin_embedding(features_neighbors).float()
        x = self.cconv(inp_features=x, inp_positions=neighbors, out_positions=out_pos, extents=2.0)
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



