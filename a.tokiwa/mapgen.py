import healpy as hp
import pandas as pd
import numpy as np
import glob

#!wget -P /gpfs02/work/akira.tokiwa/gpgpu/data -c https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt

def input_process(cl_dir):
    input_cl = pd.read_csv(cl_dir,delim_whitespace=True, index_col=0)
    lmax = input_cl.index[-1]
    cl = input_cl.divide(input_cl.index * (input_cl.index+1) / (np.pi*2), axis="index")
    cl /= 1e12
    cl = cl.reindex(np.arange(0, lmax+1))
    cl = cl.fillna(0)
    return cl, lmax

def gen_alm(low_nside, cl, lmax, seed, alm_dir):
    lclip = 3*low_nside - 1
    clipping_indices = []
    for m in range(lclip+1):
        clipping_indices.append(hp.Alm.getidx(lmax, np.arange(m, lclip+1), m))
    clipping_indices = np.concatenate(clipping_indices)
    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    alm_clipped = [each[clipping_indices] for each in alm]
    hp.write_alm(alm_dir + "Planck_bestfit_alm_seed_{}_lmax_{}_K_CMB.fits".format(seed, lclip), alm_clipped, overwrite=True)
    return alm_clipped, lclip
    
def main():
    base_dir = "/Users/akiratokiwa/workspace/SR-SPHERE/data/"
    n_gen = 1000
    cl_dir = base_dir + "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
    alm_dir = base_dir + "HR/alm/"
    map_dir = base_dir + "HR/map/"
    low_nside = 64
    
    cl, lmax = input_process(cl_dir)
    for i in range(n_gen):
        print(i)
        seed = i
        np.random.seed(seed)
        alm, lclip = gen_alm(low_nside, cl, lmax, seed, alm_dir)
        generated = hp.alm2map(alm, nside=low_nside)
        hp.write_map(map_dir+f"map_nside_{low_nside}_from_alm_seed_{seed}_lmax_{lclip}_K_CMB.fits", generated, overwrite=True)
    
    lrmap_dir = base_dir + "LR/map/"
    hrmaps = sorted(glob.glob(map_dir+"*.fits"))
    nside_lr = 32
    for f in hrmaps:
        print(f)
        hrmap = hp.read_map(f)
        lrmap = hp.ud_grade(hrmap, nside_lr)
        nside_hr = f.split("_")[2]
        seed = f.split("_")[6]
        hp.write_map(lrmap_dir+f"downsampled_map_nside_{nside_lr}_from_nside_{nside_hr}_seed_{seed}.fits", lrmap, overwrite=True)
    return 0

if __name__ == '__main__':
    main()