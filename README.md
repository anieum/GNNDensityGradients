# GNNDensityGradients
Predicting density gradients in SPH simulations with graph neural networks.

## Examples from the training process
Some of the current results of the network.


Hyperparameter search: Train loss vs. val loss

<img src="utils/images/first_hparam_search_train_loss_step.svg" width="300" />
<img src="utils/images/first_hparam_search_val_loss_step.svg" width="300" />

This is after 30 epochs.

### Predicting the density gradient (Before and after training; val dataset)
![](utils/images/eval_start.png)
![](utils/images/eval_end.png)

### Predicting the density gradient (Before and after training; train dataset)
![](utils/images/train_start.png)
![](utils/images/train_end.png)

### Each particle and their density gradient prediction. (Before and after training; val and train dataset)
![](utils/images/eval_start_2d.png)
![](utils/images/eval_end_2d.png)
![](utils/images/train_start_2d.png)
![](utils/images/train_end_2d.png)

### Loss
![](utils/images/loss.png)

Loss after 30 epochs.