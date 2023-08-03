#!/bin/bash
#SBATCH --job-name=resUnet_ploss
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/resUnet_ploss.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/slurm_log/resUnet_ploss.err  
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2 # where X is the GPU id of an available GPU
cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run
/gpfs02/work/akira.tokiwa/gpgpu/anaconda3/envs/pylit/bin/python ./run_ploss.py --loss_fn ploss --batch_size 128 --model_name resUnet_ploss