
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import open3d.ml.torch as ml3d


class CConvModel(pl.LightningModule):
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
        features_neighbors, neighbors, out_pos = x

        # Todo: only pass the neighbors features once
        features_neighbors = features_neighbors[0]
        neighbors = neighbors[0]

        # x = self.cconv(x, neighbors, out_points, extents=2.0)
        
        # Todo: I need the features for all neighbors! The outpositions are the only things that change
        # So ensure features has the same batch dimension as neighbors
        # While out_points can be of the size of the batch

        x = self.lin_embedding(features_neighbors).float()
        x = self.cconv(inp_features=x, inp_positions=neighbors, out_positions=out_pos, extents=2.0)
        x = self.lin_out_mapping(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2) #todo: lr=1e-3
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)



