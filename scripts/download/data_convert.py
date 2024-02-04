import numpy as np
import healpy as hp

import argparse
from sklearn import base

def main(filename, output_filename):
    # input file
    skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
    load_blocks = [skip[i+1]-skip[i] for i in range(0, 6)]

    with open(filename, 'rb') as f:
        rec = np.fromfile(f, dtype='uint32', count=1)[0]
        nside = np.fromfile(f, dtype='int32', count=1)[0]
        npix = np.fromfile(f, dtype='int64', count=1)[0]
        rec = np.fromfile(f, dtype='uint32', count=1)[0]
        print("nside:{} npix:{}".format(nside, npix))

        rec = np.fromfile(f, dtype='uint32', count=1)[0]

        kappa = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype='float32', count=blocks)
            np.fromfile(f, dtype='uint32', count=2)
            kappa = np.append(kappa, load)
            r = r-blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks)-1:
                load = np.fromfile(f, dtype='float32', count=r)
                np.fromfile(f, dtype='uint32', count=2)
                kappa = np.append(kappa, load)

        gamma1 = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype='float32', count=blocks)
            np.fromfile(f, dtype='uint32', count=2)
            gamma1 = np.append(gamma1, load)
            r = r-blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks)-1:
                load = np.fromfile(f, dtype='float32', count=r)
                np.fromfile(f, dtype='uint32', count=2)
                gamma1 = np.append(gamma1, load)

        gamma2 = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype='float32', count=blocks)
            np.fromfile(f, dtype='uint32', count=2)
            gamma2 = np.append(gamma2, load)
            r = r-blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks)-1:
                load = np.fromfile(f, dtype='float32', count=r)
                np.fromfile(f, dtype='uint32', count=2)
                gamma2 = np.append(gamma2, load)

        omega = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype='float32', count=blocks)
            np.fromfile(f, dtype='uint32', count=2)
            omega = np.append(omega, load)
            r = r-blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks)-1:
                load = np.fromfile(f, dtype='float32', count=r)
                np.fromfile(f, dtype='uint32', count=2)
                omega = np.append(omega, load)


    print('loading completed')

    # print data
    #for i in range (npix):  
        #print(i, kappa[i], gamma1[i], gamma2[i], omega[i])

    # example of saving data as a fits file
    hp.fitsfunc.write_map(output_filename, kappa)


if __name__ == '__main__':
    base_dir = '/gpfs02/work/akira.tokiwa/gpgpu/data/WLmap_hirosaki/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='input filename')

    args = parser.parse_args()
    main(base_dir+args.filename, base_dir+args.filename.replace('.dat', '.fits'))