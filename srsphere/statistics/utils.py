
import numpy as np
import healpy as hp
import torch

def t2hpr(x):
    x_hp = hp.pixelfunc.reorder(x.cpu().detach().numpy().reshape(-1, x.shape[0]*x.shape[1])[0], n2r=True)
    return x_hp

def powerspec(model, lr, hr, inverse_transforms, n_maps, patch_size, device, lmax, result_dir=None, save_first=False, save_mapdir=None):
    input_cl =[]
    output_cl =[]
    target_cl =[]
    if result_dir is None:
        raise ValueError('result_dir must be specified')
    for i in range(n_maps):
        x = lr[patch_size*i:patch_size*(i+1)].to(device)
        y = hr[patch_size*i:patch_size*(i+1)].to(device)
        model.eval()
        with torch.inference_mode():
            y_hat = model(x)
        loss = model.loss_fn(y_hat,y)
        print(f'loss: {loss}', flush=True)
        x = t2hpr(inverse_transforms(x))
        y = t2hpr(inverse_transforms(y))
        y_hat = t2hpr(inverse_transforms(y_hat))
        if save_first and i == 0:
            if save_mapdir is None:
                raise ValueError('save_dir must be specified')
            hp.fitsfunc.write_map(save_mapdir + 'input.fits', x, overwrite=True, dtype=np.float64)
            hp.fitsfunc.write_map(save_mapdir + 'output.fits', y_hat, overwrite=True, dtype=np.float64)
            hp.fitsfunc.write_map(save_mapdir + 'target.fits', y, overwrite=True, dtype=np.float64)
        input_cl.append(hp.sphtfunc.anafast(x, lmax=lmax))
        output_cl.append(hp.sphtfunc.anafast(y_hat, lmax=lmax))
        target_cl.append(hp.sphtfunc.anafast(y, lmax=lmax))
    input_cl_mean = np.mean(input_cl, axis=0)
    output_cl_mean = np.mean(output_cl, axis=0)
    target_cl_mean = np.mean(target_cl, axis=0)
    input_cl_std = np.std(input_cl, axis=0)
    output_cl_std = np.std(output_cl, axis=0)
    target_cl_std = np.std(target_cl, axis=0)
    
    # Save the power spectrum
    np.savetxt(result_dir + 'input_cl.txt', input_cl_mean)
    np.savetxt(result_dir + 'output_cl.txt', output_cl_mean)
    np.savetxt(result_dir + 'target_cl.txt', target_cl_mean)
    np.savetxt(result_dir + 'input_cl_std.txt', input_cl_std)
    np.savetxt(result_dir + 'output_cl_std.txt', output_cl_std)
    np.savetxt(result_dir + 'target_cl_std.txt', target_cl_std)
    
    return [input_cl, output_cl, target_cl]