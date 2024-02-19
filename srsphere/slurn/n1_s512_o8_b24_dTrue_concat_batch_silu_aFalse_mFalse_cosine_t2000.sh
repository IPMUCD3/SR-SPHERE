#!/bin/bash
#SBATCH --job-name=n1_s512_o8_b24_dTrue_concat_batch_silu_aFalse_mFalse_cosine_t2000
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/slurn/log/n1_s512_o8_b24_dTrue_concat_batch_silu_aFalse_mFalse_cosine_t2000.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/slurn/log/n1_s512_o8_b24_dTrue_concat_batch_silu_aFalse_mFalse_cosine_t2000.err  
#SBATCH --time=999:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

source /home/akira.tokiwa/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5 # where X is the GPU id of an available GPU

conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE
python -m train --n_maps 1 --nside 512 --order 8 --batch_size 24 --difference True --conditioning concat --norm_type batch --act_type silu --use_attn False --mask False --scheduler cosine --timesteps 2000 --log_name n1_s512_o8_b24_dTrue_concat_batch_silu_aFalse_mFalse_cosine_t2000