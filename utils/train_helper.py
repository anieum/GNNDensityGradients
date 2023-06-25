from pytorch_lightning.tuner import Tuner
from utils.visualization import fig_to_tensor

def find_learning_rate(trainer, model, datamodule):
    lr_finder = Tuner(trainer).lr_find(model=model, datamodule=datamodule)

    fig = fig_to_tensor(lr_finder.plot(suggest=True))
    tensorboard = trainer.logger.experiment
    tensorboard.add_image(f"results/lr_finder", fig, global_step=trainer.global_step)

    return lr_finder.suggestion()