import os
import glob
import argparse
import tqdm
import healpy as hp
import pandas as pd
import numpy as np

# Constants
DEFAULT_NSIDE_HR = 64
DEFAULT_NSIDE_LR = 32


def input_process(cl_dir):
    """
    Process input power spectrum data.
    """
    input_cl = pd.read_csv(cl_dir, delim_whitespace=True, index_col=0)
    lmax = input_cl.index[-1]
    cl = input_cl.divide(input_cl.index * (input_cl.index + 1) / (2 * np.pi), axis="index")
    cl /= 1e12
    cl = cl.reindex(np.arange(0, lmax + 1))
    cl = cl.fillna(0)
    return cl, lmax


def gen_alm(low_nside, cl, lmax, seed, alm_dir):
    """
    Generate alm and save to file.
    """
    lclip = 3 * low_nside - 1
    clipping_indices = []
    for m in range(lclip + 1):
        clipping_indices.append(hp.Alm.getidx(lmax, np.arange(m, lclip + 1), m))
    clipping_indices = np.concatenate(clipping_indices)
    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    alm_clipped = [each[clipping_indices] for each in alm]
    alm_filename = f"Planck_bestfit_alm_seed_{seed}_lmax_{lclip}_K_CMB.fits"
    hp.write_alm(os.path.join(alm_dir, alm_filename), alm_clipped, overwrite=True)
    return alm_clipped, lclip


def generate_map_and_save(alm, nside, seed, lclip, map_dir):
    """
    Generate map from alm and save to file.
    """
    generated = hp.alm2map(alm, nside=nside)
    map_filename = f"map_nside_{nside}_from_alm_seed_{seed}_lmax_{lclip}_K_CMB.fits"
    hp.write_map(os.path.join(map_dir, map_filename), generated, overwrite=True, dtype=np.float64)


def downsample_and_save(hrmaps, nside_lr, lrmap_dir):
    """
    Downsample high resolution maps and save to file.
    """
    for f in hrmaps:
        hrmap = hp.read_map(f)
        lrmap = hp.ud_grade(hrmap, nside_lr)
        nside_hr = f.split("_")[2]
        seed = f.split("_")[6]
        downsampled_map_filename = f"downsampled_map_nside_{nside_lr}_from_nside_{nside_hr}_seed_{seed}.fits"
        hp.write_map(os.path.join(lrmap_dir, downsampled_map_filename), lrmap, overwrite=True, dtype=np.float64)


def main(datadir, n_gen, nside_hr=DEFAULT_NSIDE_HR, nside_lr=DEFAULT_NSIDE_LR):
    """
    Main function to generate maps.
    """
    cl_dir = os.path.join(datadir, "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
    alm_dir = os.path.join(datadir, "HR/alm/")
    map_dir = os.path.join(datadir, "HR/map/")
    lrmap_dir = os.path.join(datadir, "LR/map/")

    # Ensure directories exist
    for directory in [alm_dir, map_dir, lrmap_dir]:
        os.makedirs(directory, exist_ok=True)

    cl, lmax = input_process(cl_dir)

    for i in tqdm.tqdm(range(n_gen)):
        seed = i
        np.random.seed(seed)
        alm, lclip = gen_alm(nside_hr, cl, lmax, seed, alm_dir)
        generate_map_and_save(alm, nside_hr, seed, lclip, map_dir)

    hrmaps = sorted(glob.glob(os.path.join(map_dir, "*.fits")))
    downsample_and_save(hrmaps, nside_lr, lrmap_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate maps with given parameters")
    parser.add_argument('--datadir', type=str, required=True, help='Directory to save the data')
    parser.add_argument('--nummaps', type=int, default=100, help='Number of maps to generate')
    args = parser.parse_args()
    main(args.datadir, args.nummaps)