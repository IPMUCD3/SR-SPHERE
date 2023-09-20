#!/bin/bash
#SBATCH --job-name=VGG_cosmo
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/test.out
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/test.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1 # where X is the GPU id of an available GPU

source /gpfs02/work/akira.tokiwa/gpgpu/anaconda3/etc/profile.d/conda.sh
conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run
python ./train_VGG_cosmo.py