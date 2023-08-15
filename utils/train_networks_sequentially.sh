#!/usr/bin/env bash
# conda activate CConv

cd ..
# trainer_dir="/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20"
trainer_dir="/home/jakob/ray_results4/LightningTrainer_2023-08-07_20-52-58"

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
# python train_network.py --name "00031_run1" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run1" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run1" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run2" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run2" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run2" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run3" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run3" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run3" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run4" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run4" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run4" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run5" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run5" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run5" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run6" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run6" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run6" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run7" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run7" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run7" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"
#
# python train_network.py --name "00031_run8" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run8" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run8" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run9" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run9" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run9" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "00031_run10" --seed 278346 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run10" --seed 147923 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_run10" --seed 393589 --params_path "${trainer_dir}/00031_31/params.json"

#python train_network.py --name "00031_random" --seed 716915 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_random" --seed 601457 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_random" --seed 144861 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_random" --seed 774658 --params_path "${trainer_dir}/00031_31/params.json"
# python train_network.py --name "00031_random" --seed 435801 --params_path "${trainer_dir}/00031_31/params.json"

# python train_network.py --name "default1"
# python train_network.py --name "default2"
# python train_network.py --name "default3"
# python train_network.py --name "default4"
# python train_network.py --name "default5"

# python train_network.py --name "default6"
# python train_network.py --name "default7"
# python train_network.py --name "default8"
# python train_network.py --name "default9"
# python train_network.py --name "default10"

python train_network.py --name "best_bayes" --checkpoint_path "${trainer_dir}/0e385726_32/rank_0/logs/srch/checkpoints/last.ckpt" --params_path "${trainer_dir}/0e385726_32/params.json"

# Step 1: Find seeds with notable initial performance differences
# Step 2: Train 10 networks each with the seeds.
# Random seeds:
# 716915 601457 144861 774658 435801