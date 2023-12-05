
from glob import glob
import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v2 as imageio
import yaml
import argparse
import pytorch_lightning as pl

from scripts.maploader.maploader import get_data_from_params, get_normalized_from_params, get_log2linear_transform
from scripts.utils.run_utils import get_parser
from scripts.utils.params import set_params

def plot_maps_png(i, sr_hp, lr_hp, hr_hp, png_dir, tmp_min, tmp_max, verbose=False):
    fig = plt.figure(figsize=(12,4))
    hp.mollview(sr_hp, nest=True, fig=fig, title=f'Generated Diff step_{str((99-i)*10).zfill(3)}+ LR', sub=(1,3,1), min=tmp_min, max=tmp_max)
    hp.mollview(hr_hp, nest=True, fig=fig, title='HR', sub=(1,3,2), min=tmp_min, max=tmp_max)
    hp.mollview(lr_hp, nest=True, fig=fig, title='LR', sub=(1,3,3), min=tmp_min, max=tmp_max)
    if verbose:
        plt.show()
    else:
        fig.savefig(png_dir + f"/step_{(99-i)*10}.png", dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    lr, hr = get_data_from_params(params)
    data_input, data_condition, transforms_lr, inverse_transforms_lr, transforms_hr, inverse_transforms_hr, range_min_lr, range_max_lr, range_min_hr, range_max_hr = get_normalized_from_params(lr, hr, params)

    map_dir = f"{args.base_dir}/results/imgs/diffusion/{params['train']['log_name']}"

    batch_size = params['train']['batch_size']*2 if batch_size is None else batch_size
    patch_size = 12 * (params['data']['order'])**2
    NUM_BATCHES = int(patch_size/batch_size)
    print(f"BATCH_SIZE: {batch_size}, NUM_BATCHES: {NUM_BATCHES}")

    diffsteps = int(params['diffusion']['timesteps'])//10
    LMAX = int(3/2 * params['data']['nside']) 
    
    lr_hp = np.hstack(inverse_transforms_lr(data_condition).detach().cpu().numpy()[:patch_size, : , 0])
    lr_sample = np.hstack(lr.detach().cpu().numpy()[:patch_size, : , 0])
    hr_hp = np.hstack(inverse_transforms_hr(data_input).detach().cpu().numpy()[:patch_size, : , 0])

    transforms_hp, inverse_transforms_hp = get_log2linear_transform()

    input_cl =hp.sphtfunc.anafast(inverse_transforms_hp(t2hpr(lr_hp)), lmax=LMAX)
    target_cl =hp.sphtfunc.anafast(inverse_transforms_hp(t2hpr(hr_hp)), lmax=LMAX) * 8

    map_diffused = read_maps(map_dir, diffsteps=diffsteps, batch_size=NUM_BATCHES)
    print(f"map_diffused.shape: {map_diffused.shape}")

    png_dir = map_dir + "/png"
    ps_dir = map_dir + "/ps" 
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(ps_dir, exist_ok=True)

    print("start plotting pngs and ps")
    for i in range(100):
        print(f"step_{(99-i)*10}")
        sr_hp = inverse_transforms_hp(lr_sample + map_diffused[99-i], range_min_hr, range_max_hr)
        output_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(sr_hp)-1), lmax=LMAX)
        cls = [input_cl, target_cl, output_cl]
        plot_ps_png(i, cls, map_dir + "/ps", LMAX)

    png_files = sorted(glob(png_dir + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))[::-1]
    ps_files = sorted(glob(ps_dir + "/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-2]))[::-1]

    png_images = []
    ps_images = []

    for png_file, ps_file in zip(png_files, ps_files):
        tmp_png = imageio.imread(png_file)
        tmp_ps = imageio.imread(ps_file)
        # keep the same size
        png_images.append(tmp_png)
        ps_images.append(tmp_ps)

    # stay at the last frame for 3 seconds
    for i in range(30):
        png_images.append(imageio.imread(png_files[-1]))
        ps_images.append(imageio.imread(ps_files[-1]))

    print("start saving gif")
    # save animation gif
    imageio.mimsave(map_dir+"diffused.gif", png_images, duration=100, loop=0)
    imageio.mimsave(map_dir+"diffused_ps.gif", ps_images, duration=100, loop=0)

if __name__ == "__main__":
    main()