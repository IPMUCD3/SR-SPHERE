
import  torch
import numpy as np
import matplotlib.pyplot as plt

def filter_boost(t, k, gamma):
    t_fourier = torch.fft.fftn(t).to(k.device)
    t_fourier = t_fourier * k**gamma
    t = torch.fft.ifftn(t_fourier).to(k.device)
    t = torch.abs(t)
    return t

def calculate_k(t):
    kx = torch.fft.fftfreq(t.shape[0])
    ky = torch.fft.fftfreq(t.shape[1])
    kz = torch.fft.fftfreq(t.shape[2])

    kx = kx.reshape(-1, 1, 1)
    ky = ky.reshape(1, -1, 1)
    kz = kz.reshape(1, 1, -1)

    k = torch.sqrt(kx**2 + ky**2 + kz**2)

    return k

def batch_filter_boost(t, k, gamma):
    """
    Apply the filter to every map in the data tensor.
    """
    filtered_data = []
    for i in range(t.shape[0]):
        if i % 1000 == 0:
            print(f"filtering {i}th map")
        filtered_map = filter_boost(t[i], k, gamma)
        filtered_data.append(filtered_map.unsqueeze(0))
    return torch.cat(filtered_data, dim=0) # shape: (n_maps, 64, 64, 64)

def plot_filter_boost(t, k, img_path):
    k = calculate_k(t).to(t.device)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for gamma in [0, 1, 2, 3]:
        filterd = filter_boost(t, k, gamma)
        ax[gamma].imshow(filterd.detach().cpu().numpy()[0], vmin=0, vmax=3*np.mean(filterd.detach().cpu().numpy()[0]))
        ax[gamma].set_title(f"gamma={gamma}")
        print(f"gamma={gamma}, mean={np.mean(filterd.detach().cpu().numpy()[0])}")
    
    fig.savefig(img_path, bbox_inches='tight', dpi=100)

