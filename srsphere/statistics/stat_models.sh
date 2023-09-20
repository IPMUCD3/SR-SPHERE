#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/statistics/log/stats.out
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/statistics/log/stats.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1 # where X is the GPU id of an available GPU

PYTHON=/gpfs02/work/akira.tokiwa/gpgpu/anaconda3/envs/pylit/bin/python

$PYTHON /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/statistics/stat_models.py
$PYTHON /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/visualization/plot_powerspec.py
$PYTHON /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/visualization/plot_sample.py
