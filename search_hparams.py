import pytorch_lightning as pl
from utils.train_helper import *
from models.cconv import CConvModel
from datasets.density_data_module import DensityDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray import air, tune, init
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
import torch.cuda

# See https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example
# and https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

# The maximum training epochs
num_epochs = 5

# Number of sampls from parameter space
num_samples = 2


hparams = {
    # "layer_1_size": tune.choice([32, 64, 128]),
    # "layer_2_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),

    # Dataset
    'dataset_dir': 'datasets/data/dpi_dam_break/train',
    'data_split': (0.7, 0.15, 0.15),
    'batch_size': 10,        # care, this is used in the model and datamodule
    'shuffle': True,
    'cache': True,            # Preprocess and preload dataset into memory
    'device': 'cuda'
}

datamodule = DensityDataModule(data_dir=hparams['dataset_dir'], batch_size=hparams['batch_size'], data_split=hparams['data_split'], shuffle=hparams['shuffle'], cache=hparams['cache'], device=hparams['device'])
logger = TensorBoardLogger("lightning_logs", name="cconv-hparam-search", version=".")
lightning_config = (
    LightningConfigBuilder()
    .module(cls=CConvModel, hparams=hparams)
    .trainer(max_epochs=num_epochs, logger=logger, accelerator="gpu", enable_progress_bar=False)
    .fit_params(datamodule=datamodule)
    .checkpointing(monitor="val_loss", mode="min", save_top_k=3)
    .build()
)
run_config = RunConfig(
    checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
)


init(num_cpus=2, num_gpus=1)
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 0.5})

# Define a base LightningTrainer without hyper-parameters for Tuner
lightning_trainer = LightningTrainer(scaling_config=scaling_config, run_config=run_config)
datamodule.setup("Initialize")
datamodule.to('cpu')
def tune_mnist_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        )
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="max")
    best_result

print("Cuda is available: ", torch.cuda.is_available())
tune_mnist_asha(num_samples=num_samples)