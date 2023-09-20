
import glob
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import argparse

def plot_sample(mapdir, stat_models=False):
    input = hp.fitsfunc.read_map(mapdir + 'input.fits')
    output = hp.fitsfunc.read_map(mapdir + 'output.fits')
    target = hp.fitsfunc.read_map(mapdir + 'target.fits')

    fig = plt.figure(figsize=(7, 9))
    # set min and max among input, output, and target
    tmp_min = round(np.min([np.min(input), np.min(output), np.min(target)]), 1)
    tmp_max = round(np.max([np.max(input), np.max(output), np.max(target)]), 1)
    hp.orthview(input, fig=fig, title=r'LR $64^3$', sub=(3, 1, 1), min=tmp_min, max=tmp_max)
    hp.orthview(output, fig=fig, title=r'SR $128^3$', sub=(3, 1, 2), min=tmp_min, max=tmp_max)
    hp.orthview(target, fig=fig, title=r'HR $128^3$', sub=(3, 1, 3), min=tmp_min, max=tmp_max)

    if stat_models==True:
        model_name = mapdir.split('/')[-2].split('_')[0]
        loss_fn = mapdir.split('/')[-2].split('_', 1)[1]
        fig.savefig(f'{mapdir}/sample_{model_name}_{loss_fn}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
def main(stat_models=False):
    if stat_models==True:
        mapdirs = glob.glob("/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/visualization/*/")
        for mapdir in mapdirs:
            plot_sample(mapdir, stat_models=True)
    else:
        raise ValueError("Not implemented")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stat_models', type=bool, default=True)
    args = args.parse_args()

    main(stat_models=args.stat_models)