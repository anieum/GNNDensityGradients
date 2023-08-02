#!/usr/bin/env bash
conda activate CConv
cd ..
# store path as variable
trainer_dir="/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20"

python train_network.py --checkpoint_path "${trainer_dir}/00031_31/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --checkpoint_path "${trainer_dir}/00087_87/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00087_87/params.json"
python train_network.py --checkpoint_path "${trainer_dir}/00030_30/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00087_87/params.json"
