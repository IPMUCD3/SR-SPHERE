#!/bin/bash
#SBATCH --job-name=diffmap_generation
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/diffmap_diff.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/diffmap_diff.err  
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

source /home/akira.tokiwa/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7 # where X is the GPU id of an available GPU

conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE
#python -m scripts.diffusion.evaluation.diffusemap --target "HR" --scheduler "linear" --order 4 --nmaps 1 --batch_size 4
#python -m scripts.diffusion.evaluation.diffusemap --target "HR" --scheduler "cosine" --order 4 --nmaps 1 --batch_size 4
python -m scripts.diffusion.evaluation.diffusemap --target "difference" --scheduler "linear" --order 4 --nmaps 1 --batch_size 4
#python -m scripts.diffusion.evaluation.diffusemap --target "difference" --scheduler "cosine" --order 4 --nmaps 1 --batch_size 4