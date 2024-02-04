#!/bin/bash
#SBATCH --job-name=fname
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/validation/fname.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/log/diffusion/validation/fname.err  
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

source /home/akira.tokiwa/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=_device # where X is the GPU id of an available GPU

conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE
python -m scripts.diffusion.evaluation.diffusemap --model _model --target _target --mask _ifmask --use_attn True --scheduler _schedule --transform_type _trans --conditioning _cond --order _order --n_maps _nmap --batch_size _batch