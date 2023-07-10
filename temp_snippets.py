# https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html

# try this instead:
# https://docs.ray.io/en/latest/tune/examples/tune-vanilla-pytorch-lightning.html#tune-vanilla-pytorch-lightning-ref



from ray.tune.logger import LoggerCallback
from ray.tune.logger import TBXLoggerCallback
from ray.air import session

# See https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example
# and https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

# TODO: for each model, track total number of parameters


from ray.tune.logger import TBXLoggerCallback
class MyLoggerCallback(TBXLoggerCallback):
    #def on_trial_result(self, iteration: int, trials: List[Trial], trial: Trial, result: Dict, **info):
    def on_trial_result(self, trial, result, *args, **kwargs):
        print("Trial result: ", result)
        #model = CConvModel(trial.config['lightning_config']['_module_init_config']['hparams'])
        #result['number_of_params'] = sum(p.numel() for p in model.parameters())
        #result['number_of_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #session.report({'number_of_params': result['number_of_params'], 'number_of_trainable_params': result['number_of_trainable_params']})
        return super().on_trial_result(trial=trial, result=result, *args, **kwargs)


# import lightning callback



def get_num_params(spec):
    """
    Track the number of hyperparameters in the model by instantiating a model at the parameter selection stage.

    Note: This is highly unsupported. It's however the only way I found to get the number of parameters into the
    hparams page of the tensorboard.

    A better way would be to write a custom logger using this callback:
    https://docs.ray.io/en/latest/tune/api/doc/ray.tune.logger.TBXLoggerCallback.html#ray.tune.logger.TBXLoggerCallback
    https://docs.ray.io/en/latest/tune/tutorials/tune-output.html#how-to-build-custom-tune-loggers
    """

    conf = spec.config.lightning_config
    model = CConvModel(conf._module_init_config['hparams'])
    number_of_params = sum(p.numel() for p in model.parameters())
    conf._module_init_config['hparams']['number_of_params'] = number_of_params
    return number_of_params

# Derived metric
# 'number_of_params' : tune.sample_from(get_num_params),
# 'number_of_trainable_params' : tune.sample_from(lambda spec: sum(p.numel() for p in CConvModel(spec.config).parameters() if p.requires_grad)),


lightning_trainer = LightningTrainer(
    scaling_config = ScalingConfig(
        num_workers          = 1,
        use_gpu              = True,
        resources_per_worker = {'CPU': 2, 'GPU': 0.5}
    ),
    run_config = RunConfig(
        checkpoint_config = CheckpointConfig(
            num_to_keep                = 2,
            checkpoint_score_attribute = 'val_loss',
            checkpoint_score_order     = 'min',
        ),
        callbacks = [MyLoggerCallback()],
    )
)