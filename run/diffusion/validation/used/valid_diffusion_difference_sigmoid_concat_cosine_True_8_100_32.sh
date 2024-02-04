#!/bin/bash
#SBATCH --job-name=valid_diffusion_difference_sigmoid_concat_cosine_True_8_100_32
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/validation/valid_diffusion_difference_sigmoid_concat_cosine_True_8_100_32.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/validation/valid_diffusion_difference_sigmoid_concat_cosine_True_8_100_32.err  
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
python -m scripts.diffusion.evaluation.diffusemap --model diffusion --target difference --mask True --use_attn True --scheduler cosine --transform_type sigmoid --conditioning concat --order 8 --n_maps 100 --batch_size 32