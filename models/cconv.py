
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import open3d.ml.torch as ml3d


class CConvModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.lin_embedding = nn.Linear(1,8) 
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
        particle_positions, x = x
        x = self.lin_embedding(x)
        x = self.cconv(x, particle_positions, particle_positions, extents=2.0)
        x = self.lin_out_mapping(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)

        pred_y = self(x)
        loss = F.mse_loss(pred_y, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)



