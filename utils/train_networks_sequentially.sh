#!/usr/bin/env bash
# conda activate CConv

cd ..
trainer_dir="/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20"

# echo "${trainer_dir}/00031_31/logs/srch/checkpoints/last.ckpt"

# python train_network.py --checkpoint_path "${trainer_dir}/00031_31/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --checkpoint_path "${trainer_dir}/00087_87/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00087_87/params.json"
# python train_network.py --checkpoint_path "${trainer_dir}/00030_30/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00030_30/params.json"
# python train_network.py --checkpoint_path "${trainer_dir}/00098_98/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00098_98/params.json"
# python train_network.py --checkpoint_path "${trainer_dir}/00076_76/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/00076_76/params.json"

# Seeds are:
# - Best: 278346
# - Median: 147923
# - Worst: 393589

# ANOVA test: initial vs. final performance
python train_network.py --name "00031_run1" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run1" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run1" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

python train_network.py --name "00031_run2" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run2" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run2" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

python train_network.py --name "00031_run3" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run3" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run3" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

python train_network.py --name "00031_run4" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run4" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run4" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

python train_network.py --name "00031_run5" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run5" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
python train_network.py --name "00031_run5" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_seedsearch" --seed 333 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_seedsearch" --seed 444 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_seedsearch" --seed 555 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_seedsearch" --seed 666 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_seedsearch" --seed 777 --params_path "${trainer_dir}/00031_31/params.json"

# Step 1: Find seeds with notable initial performance differences
# Step 2: Train 10 networks each with the seeds.
