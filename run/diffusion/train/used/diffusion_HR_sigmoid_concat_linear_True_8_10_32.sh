#!/bin/bash
#SBATCH --job-name=diffusion_HR_sigmoid_concat_linear_True_8_10_32
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/train/diffusion_HR_sigmoid_concat_linear_True_8_10_32.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/train/diffusion_HR_sigmoid_concat_linear_True_8_10_32.err  
#SBATCH --time=99:00:00
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
python -m run.diffusion.run_diffusion --model diffusion --target HR --mask True --use_attn False --scheduler linear --transform_type sigmoid --conditioning concat --order 8 --n_maps 10 --batch_size 32