#!/bin/bash
#SBATCH --job-name=resUnet_l1
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/threeconv_mse.out
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/threeconv_mse.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0 # where X is the GPU id of an available GPU
cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run
/gpfs02/work/akira.tokiwa/gpgpu/anaconda3/envs/pylit/bin/python ./train_models.py --batch_size 32 --model threeconv --loss_fn mse