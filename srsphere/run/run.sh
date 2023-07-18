#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/log/%j.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/log/%j.err  
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4 # where X is the GPU id of an available GPU

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE-1/srsphere/run/
/gpfs02/work/akira.tokiwa/gpgpu/anaconda3/envs/pylit/bin/python ./run_ResUnet.py