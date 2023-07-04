import pytorch_lightning as pl
from utils.train_helper import *
from models.cconv import CConvModel
from datasets.density_data_module import DensityDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray import air, tune
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
    'cache': False,            # Preprocess and preload dataset into memory
    'device': 'cuda'
}

datamodule = DensityDataModule(data_dir=hparams['dataset_dir'], batch_size=hparams['batch_size'], data_split=hparams['data_split'], shuffle=hparams['shuffle'], cache=hparams['cache'], device=hparams['device'])
logger = TensorBoardLogger("lightning_logs", name="cconv-hparam-search", version=".")

# DO NOT CACHE THE DATAMODULE AND PASS IT DIRECTLY WITHOUT LOADERS.
# OTHERWISE RAY TUNE WILL SERIALIZE THE ENTIRE DATASET AND BLOW UP MEMORY AND DISK SPACE
datamodule.setup("fit")
torch.cuda.empty_cache()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


lightning_config = (
    LightningConfigBuilder()
    .module(cls=CConvModel, hparams=hparams)
    .trainer(max_epochs=num_epochs, logger=logger, accelerator="gpu", enable_progress_bar=False)
    .fit_params(train_dataloaders=train_loader, val_dataloaders=val_loader)
    .checkpointing(monitor="val_loss", mode="min", save_top_k=2)
    .build()
)

ray.shutdown()
ray.init(num_cpus=6, num_gpus=1)

lightning_trainer = LightningTrainer(
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 0.5}
    ),
    run_config = RunConfig(
        checkpoint_config = CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
)

def tune_models(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    # Todo: search algorithm (the standard is either hyperopt, grid or random search)
    # https://docs.ray.io/en/latest/tune/api/suggestion.html

    tuner = tune.Tuner(
        lightning_trainer,
        param_space = {"lightning_config": lightning_config},
        tune_config = tune.TuneConfig(
            metric      = "val_loss",
            mode        = "min",
            num_samples = num_samples,
            scheduler   = scheduler,
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="max")
    best_result

print("Cuda is available: ", torch.cuda.is_available())
tune_models(num_samples=num_samples)