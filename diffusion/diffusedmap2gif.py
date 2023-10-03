
from glob import glob
import healpy as hp
import matplotlib.pyplot as plt
import re
import numpy as np
import imageio.v2 as imageio
import yaml
import sys

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE')
from srsphere.data.maploader import MapDataset, get_datasets, transform_combine

map_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/diffusion/result/ani/"
map_files = glob(map_dir + "diffused_step*.fits")
map_files = sorted(map_files, key=lambda s: int(re.findall(r'\d+', s)[1]), reverse=True)
input = hp.fitsfunc.read_map(f'{map_dir}../img/input.fits')
target = hp.fitsfunc.read_map(f'{map_dir}../img/target.fits')
input_cl = hp.sphtfunc.anafast(input, lmax=64*4)
target_cl = hp.sphtfunc.anafast(target, lmax=64*4)



def get_minmax():
    config_file = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/diffusion/params.yaml"
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    data_lr = MapDataset(config_dict['data']['lrmaps_dir'], config_dict['data']['n_maps'],  config_dict['data']['nside_lr'],config_dict['data']['order'], config_dict['data']['issplit'], config_dict['data']["normalize"]).__getitem__(0)
    data_hr = MapDataset(config_dict['data']['hrmaps_dir'], config_dict['data']['n_maps'],  config_dict['data']['nside_hr'],config_dict['data']['order'], config_dict['data']['issplit'], config_dict['data']["normalize"]).__getitem__(0)
    RANGE_MIN, RANGE_MAX = (data_hr-data_lr).min().clone().detach().numpy(), (data_hr-data_lr).max().clone().detach().numpy()
    return RANGE_MIN, RANGE_MAX

def invt(x, rangemin, rangemax):
    return (x + 1) * (rangemax - rangemin) / 2 + rangemin

RANGE_MIN, RANGE_MAX = get_minmax()
input = invt(input, RANGE_MIN, RANGE_MAX) 
target = invt(target, RANGE_MIN, RANGE_MAX)

def t2hpr(x):
    x_hp = hp.pixelfunc.reorder(x, n2r=True)
    return x_hp

def plot_sample(fname, step, input, output, target):
    fig = plt.figure(figsize=(7, 9))
    # set min and max among input, output, and target
    tmp_min = round(np.min(target), 1)
    tmp_max = round(np.max(target), 1)
    hp.orthview(input, fig=fig, title=r'LR $64^3$', sub=(2, 2, 1), min=tmp_min, max=tmp_max)
    hp.orthview(output, fig=fig, title='SR: step={}'.format(step), sub=(2, 1, 2), min=tmp_min, max=tmp_max)
    hp.orthview(target, fig=fig, title=r'HR $128^3$', sub=(2, 2, 2), min=tmp_min, max=tmp_max)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)

def plot_powerspec(cls, fname, step):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r"LR $64^3$", r"HR $128^3$", 'SR: step={}'.format(step), "noise (HR-LR): step={}".format(step)]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    ax.set_title("Power spectrum, SR: step={}".format(step), fontsize=12)
    for i, cl in enumerate(cls):
        if i == 0:
            ell = np.arange(len(cl))
        tmp_ps = (ell*(ell+1)*cl/(2*np.pi))
        ax.plot(ell, tmp_ps, label=labels[i], color=colors[i])
        
    ax.set_xlabel("l", fontsize=16)
    ax.set_ylabel(r"$\mathrm{l(l+1)C_{l}/2\pi}\;\; $", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=12, loc="lower right")    
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)

img_flag = True
ps_flag = True
# read map and plot mollview, and save all mollview as animation gif
images = []
images_ps = []
for f in map_files:
    print(f)
    dmap = invt(hp.read_map(f), RANGE_MIN, RANGE_MAX)
    output = input + t2hpr(dmap)
    output_cl = hp.sphtfunc.anafast(output, lmax=64*4)
    noise_cl = hp.sphtfunc.anafast(t2hpr(dmap), lmax=64*4)
    cls = [input_cl, target_cl, output_cl, noise_cl]
    step = int(500 - int(f.rsplit("step", 1)[1].split(".")[0]))
    if img_flag:
        fname = map_dir + "png/diffused_step{}.png".format(step)
        plot_sample(fname, step, input, output, target)
    images.append(imageio.imread(map_dir+"png/diffused_step{}.png".format(step)))
    if ps_flag:
        fname = map_dir + "png/diffused_step{}_ps.png".format(step)
        plot_powerspec(cls, fname, step)
    images_ps.append(imageio.imread(map_dir+"png/diffused_step{}_ps.png".format(step)))
# stop the animation at the last frame for 2 sec (append the last frame 100 times)
for i in range(50):
    images.append(imageio.imread(map_dir+"png/diffused_step{}.png".format(step)))
    images_ps.append(imageio.imread(map_dir+"png/diffused_step{}_ps.png".format(step)))
    
# save animation gif
imageio.mimsave(map_dir+"../diffused.gif", images, duration=100, loop=0)
imageio.mimsave(map_dir+"../diffused_ps.gif", images_ps, duration=100, loop=0)