
from glob import glob
import healpy as hp
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v2 as imageio
import yaml

from scripts.maploader.maploader import get_data, get_minmaxnormalized_data

def read_maps(map_dir, diffsteps=100, batch_size=4):
    maps = sorted(glob(map_dir + "/*.npy"), key=lambda x: (int(x.split("/")[-1].split("_")[2]), int(x.split("/")[-1].split(".")[0].split("_")[-1])))
    map_diffused = []
    for i in range(diffsteps):
        map_steps = []
        for j in range(batch_size):
            #print("Current map: {}".format(maps[i*batch_size+j]))
            map_steps.append(np.load(maps[i*batch_size+j]))
        map_steps = np.array(map_steps)
        map_steps = np.hstack(map_steps)
        map_diffused.append(map_steps)
    map_diffused = np.array(map_diffused)
    return map_diffused

def t2hpr(x):
    x_hp = hp.pixelfunc.reorder(x, n2r=True)
    return x_hp

def inverse_transforms_hp(x, range_min, range_max):
    x = (x + 1) / 2 * (range_max - range_min) + range_min
    return x

def plot_ps(cls, fig, ax):
    if len(cls) == 2:
        labels = ["input", "target"]
    elif len(cls) == 3:
        labels = ["input", "target", "output"]
    elif len(cls) == 4:
        labels = ["input", "target", "output", "diff"]
    else:
        raise ValueError("cls must be 2 or 3 or 4 length")
    ell = np.arange(len(cls[0]))
    for cl, label in zip(cls, labels):
        ax.plot(ell*(ell+1)*cl/(2*np.pi), label=label, alpha=0.7)
    ax.set_xlabel("l", fontsize=12)
    ax.set_ylabel(r"$l(l+1)C_{l}/2\pi\;\; $", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=12)
    return fig, ax

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

def plot_ps_png(i, cls, ps_dir, lmax, verbose=False):
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    fig, ax = plot_ps(cls, fig, ax)
    ax.set_title(f"step_{str((99-i)*10).zfill(3)}")
    if verbose:
        plt.show()
    else:
        fig.savefig(ps_dir + f"/step_{(99-i)*10}_ps.png", dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    ckpt_dir = f"{base_dir}/ckpt_logs/diffusion/cosine_o4_b24/version_1"
    with open(f"{ckpt_dir}/hparams.yaml", 'r') as stream:
        config_dict = yaml.safe_load(stream)
    map_dir = f"{base_dir}/results/imgs/diffusion/{config_dict['train']['log_name']}"

    BATCH_SIZE = 48#config_dict['train']['batch_size']
    PATCH_SIZE = 12 * (config_dict['data']['order'])**2
    NUM_BATCHES = PATCH_SIZE//BATCH_SIZE
    print(f"BATCH_SIZE: {BATCH_SIZE}, PATCH_SIZE: {PATCH_SIZE}, NUM_BATCHES: {NUM_BATCHES}")
    
    NSIDE = config_dict['data']['nside']
    diffsteps = int(config_dict['diffusion']['timesteps'])//10
    LMAX = NSIDE*3 
    print(f"NSIDE: {NSIDE}, LMAX: {LMAX}")

    lr = get_data(config_dict['data']['LR_dir'], config_dict['data']['n_maps'], config_dict['data']['nside'], config_dict['data']['order'], issplit=True)
    hr = get_data(config_dict['data']['HR_dir'], config_dict['data']['n_maps'], config_dict['data']['nside'], config_dict['data']['order'], issplit=True)

    hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    # range_min_hr, range_max_hr  to float]
    range_min_hr, range_max_hr = float(range_min_hr.detach().cpu().numpy()), float(range_max_hr.detach().cpu().numpy())
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))
    print("HR data normalized. min: {}, max: {}".format(hr.min(), hr.max()))

    lr = transforms_hr(lr)
    print("LR data normalized by HR range. min: {}, max: {}".format(lr.min(), lr.max()))
    
    lr_hp = np.hstack(inverse_transforms_hr(lr).detach().cpu().numpy()[:PATCH_SIZE, : , 0])
    lr_sample = np.hstack(lr.detach().cpu().numpy()[:PATCH_SIZE, : , 0])
    hr_hp = np.hstack(inverse_transforms_hr(hr).detach().cpu().numpy()[:PATCH_SIZE, : , 0])
    
    input_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(lr_hp)-1), lmax=LMAX)
    target_cl =hp.sphtfunc.anafast(np.exp(np.log(10)*t2hpr(hr_hp)-1), lmax=LMAX) * 8

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