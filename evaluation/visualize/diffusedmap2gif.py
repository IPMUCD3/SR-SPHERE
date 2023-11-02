
from glob import glob
import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v2 as imageio
import yaml
import torch
import torch.utils.data as data

from scripts.utils.run_utils import initialize_config
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data

def read_maps(map_dir, diffsteps=100, batch_size=16):
    maps = sorted(glob(map_dir + "*.npy"), key=lambda x: (int(x.split("/")[-1].split("_")[2]), int(x.split("/")[-1].split(".")[0].split("_")[-1])))
    map_diffused = []
    for i in range(diffsteps):
        map_steps = []
        for j in range(batch_size):
            map_steps.append(np.load(maps[i*batch_size+j]))
        map_steps = np.array(map_steps)
        map_steps = np.hstack(map_steps)
        map_diffused.append(map_steps)
    map_diffused = np.array(map_diffused)
    return map_diffused

def inverse_transforms_hp(x, range_min, range_max):
    if type(range_max) == torch.Tensor:
        range_max = range_max.item()
    if type(range_min) == torch.Tensor:
        range_min = range_min.item()
    x = (x + 1) / 2 * (range_max - range_min) + range_min
    return x

def t2hpr(x):
    x_hp = hp.pixelfunc.reorder(x, n2r=True)
    return x_hp

def plot_ps(cls, fig, ax):
    if len(cls) == 2:
        labels = ["input", "target"]
    elif len(cls) == 3:
        labels = ["input", "output", "target"]
    else:
        raise ValueError("cls must be 2 or 3 length")
    ell = np.arange(len(cls[0]))
    for cl, label in zip(cls, labels):
        ax.plot(ell*(ell+1)*cl/(2*np.pi), label=label)
    ax.set_xlabel("l", fontsize=16)
    ax.set_ylabel(r"$l(l+1)C_{l}/2\pi\;\; $", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=16)
    return fig, ax

def plot_maps_png(i, map_diffused, lr_sample, original_sample, png_dir, tmp_min, tmp_max):
    fig = plt.figure(figsize=(12,4))
    hp.mollview(map_diffused[99-i]+ lr_sample, nest=True, fig=fig, title=f'Generated Diff step_{str((99-i)*10).zfill(3)}+ LR', sub=(1,3,1), min=tmp_min, max=tmp_max)
    hp.mollview(original_sample + lr_sample, nest=True, fig=fig, title='HR', sub=(1,3,2), min=tmp_min, max=tmp_max)
    hp.mollview(lr_sample, nest=True, fig=fig, title='LR', sub=(1,3,3), min=tmp_min, max=tmp_max)
    fig.savefig(png_dir + f"step_{(99-i)*10}.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_ps_png(i, map_diffused, lr_sample, input_cl, target_cl, ps_dir, lmax):
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    output_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(map_diffused[99-i]+ lr_sample)-1), lmax=lmax)
    fig, ax = plot_ps([input_cl, output_cl, target_cl], fig, ax)
    ax.set_title(f"step_{str((99-i)*10).zfill(3)}")
    fig.savefig(ps_dir + f"step_{(99-i)*10}_ps.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    map_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/results/imgs/diffusion/HR_normalized/"
    config_file = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/config/config_diffusion.yaml"
    config_dict = initialize_config(config_file)

    ### get training data
    config_dict['data']['lrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    config_dict['data']['hrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    config_dict['data']['nside_lr'] = 512
    config_dict['data']['nside_hr'] = 512
    config_dict['data']["normalize"] = False
    config_dict['data']['order'] = 4
    config_dict['train']['batch_size'] = 48

    BATCH_SIZE = config_dict['train']['batch_size']
    PATCH_SIZE = 12 * (config_dict['data']['order'])**2
    NUM_BATCHES = PATCH_SIZE//BATCH_SIZE
    NSIDE = config_dict['data']['nside_lr']
    diffsteps = int(config_dict['diffusion']['timesteps'])//10
    LMAX = NSIDE*3 

    lr = get_data(config_dict['data']['lrmaps_dir'], config_dict['data']['n_maps'], config_dict['data']['nside_lr'], config_dict['data']['order'], issplit=True)
    hr = get_data(config_dict['data']['hrmaps_dir'], config_dict['data']['n_maps'], config_dict['data']['nside_hr'], config_dict['data']['order'], issplit=True)

    #lr, transforms_lr, nverse_transforms_lr, range_min_lr, range_max_lr = get_minmaxnormalized_data(lr)
    #print("LR data loaded. min: {}, max: {}".format(range_min_lr, range_max_lr))

    hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))

    lr = transforms_hr(lr)
    print("LR data normalized by HR range. min: {}, max: {}".format(lr.min(), lr.max()))
    
    lr_hp = np.hstack(inverse_transforms_hr(lr).detach().cpu().numpy()[:PATCH_SIZE, : , 0])
    hr_hp = np.hstack(inverse_transforms_hr(hr).detach().cpu().numpy()[:PATCH_SIZE, : , 0])
    
    input_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(lr_hp)-1), lmax=LMAX)
    target_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(hr_hp)-1), lmax=LMAX) * 8

    map_diffused = read_maps(map_dir, diffsteps=diffsteps, batch_size=NUM_BATCHES)

    png_dir = map_dir + "png/"
    ps_dir = map_dir + "ps/" 
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(ps_dir, exist_ok=True)

    print("start plotting pngs and ps")
    for i in range(diffsteps):
        print(f"step_{(99-i)*10}")
        plot_maps_png(i, map_diffused, lr_hp, hr_hp, png_dir, range_min_hr, range_max_hr)
        plot_ps_png(i, map_diffused, lr_hp, input_cl, target_cl, ps_dir, LMAX)

    png_files = sorted(glob(png_dir + "*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))[::-1]
    ps_files = sorted(glob(ps_dir + "*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-2]))[::-1]

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