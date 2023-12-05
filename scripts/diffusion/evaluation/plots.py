
import numpy as np
import matplotlib.pyplot as plt

def plot_ps(cls, title=None, save_dir=None, labels=None, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(8,4))

    if (labels is None)&(len(cls)<=4):
        labels = ["input", "target", "output", "diff"][:len(cls)]
    else:
        raise ValueError("labels must be given")

    ell = np.arange(len(cls[0]))
    for cl, label in zip(cls, labels):
        ax.plot(ell*(ell+1)*cl/(2*np.pi), label=label, alpha=0.7)
    ax.set_xlabel("l", fontsize=12)
    ax.set_ylabel(r"$l(l+1)C_{l}/2\pi\;\; $", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)
    if save_dir is not None:
        fig.savefig(save_dir, dpi=200, bbox_inches='tight', pad_inches=0.1)
    return fig, ax