
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_cl(cl_dir):
    input_cl = np.loadtxt(f'{cl_dir}/input_cl.txt')
    output_cl = np.loadtxt(f'{cl_dir}/output_cl.txt')
    target_cl = np.loadtxt(f'{cl_dir}/target_cl.txt')
    cls = [input_cl, output_cl, target_cl]

    input_cl_std = np.loadtxt(f'{cl_dir}/input_cl_std.txt')
    output_cl_std = np.loadtxt(f'{cl_dir}/output_cl_std.txt')
    target_cl_std = np.loadtxt(f'{cl_dir}/target_cl_std.txt')
    cl_stds = [input_cl_std, output_cl_std, target_cl_std]

    return cls, cl_stds

def plot_powerspec(cls, cl_stds, cl_dir, stat_models=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r"LR $64^3$", r"SR $128^3$", r"HR $128^3$"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax.set_title(r"Power spectrum, $N_\mathrm{side}=64$", fontsize=12)
    for i, cl in enumerate(cls):
        if i == 0:
            ell = np.arange(len(cl))
        tmp_ps = (ell*(ell+1)*cl/(2*np.pi))
        tmp_std = (ell*(ell+1)*cl_stds[i]/(2*np.pi))
        ax.fill_between(ell, (tmp_ps-tmp_std), (tmp_ps+tmp_std), alpha=0.3, color=colors[i])
        ax.plot(ell, tmp_ps, label=labels[i], color=colors[i])
        
    ax.set_xlabel("l", fontsize=16)
    ax.set_ylabel(r"$\mathrm{l(l+1)C_{l}/2\pi}\;\; $", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=12)

    if stat_models==True:
        model_name = cl_dir.split('/')[-2].split('_')[0]
        loss_fn = cl_dir.split('/')[-2].split('_', 1)[1]
        fig.savefig(f'{cl_dir}/powerspec_{model_name}_{loss_fn}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        raise ValueError("Not implemented")
    
def main(stat_models=False):
    if stat_models==True:
        cl_dirs = glob.glob("/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/statistics/result/*/")
        for cl_dir in cl_dirs:
            cls, cl_stds = read_cl(cl_dir)
            plot_powerspec(cls, cl_stds, cl_dir, stat_models=True)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stat_models', type=bool, default=True)
    args = args.parse_args()
    main(stat_models=args.stat_models)
        