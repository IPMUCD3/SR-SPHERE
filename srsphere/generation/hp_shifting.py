
import numpy as np
import healpy as hp
import math

class NestGridShift:
    """
    Implements grid shifting operations for the HEALPix NESTED pixel ordering scheme.
    """

    def __init__(self, nside, order):
        """
        Initializes the NestGridShift object.

        Args:
            nside: The resolution parameter (NSIDE) of the HEALPix map.
            order: The resolution parameter (NSIDE) of a patch.
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.order = order
        self.base_pix_per_chunk = 4 ** (int(math.log2(nside)) - int(math.log2(order)) - 1)
        self.max_shift = int(np.sqrt(nside ** 2 // self.base_pix_per_chunk)) - 1
        self.max_patch = 12

        self.shifts1, self.shifts2 = self._prepare_shifts()
        self.shift_matrix = self._prepare_shift_matrix()

    def count2shift(self, count, base_pix, direction=1):
        bin_count = bin(count)[2:]
        shift = 0
        for j in range(1, len(bin_count)+1):
            if bin_count[-j] == '1':
                shift += direction * 4 ** j * base_pix
        return shift

    def _prepare_shifts(self):
        """Calculates horizontal and vertical shift values."""
        shifts1 = []
        shifts2 = []
        for i in range(self.max_shift):
            shifts1.append(self.count2shift(i, self.base_pix_per_chunk, 1))
            shifts2.append(self.count2shift(i, self.base_pix_per_chunk, 2))
        
        return shifts1, shifts2
    
    def idx_shift1(self, count):
        shifts = np.zeros(4)
        c2 = count // 2
        if count % 2 == 0:
            shifts[0] = self.shifts1[c2]
            shifts[1] = self.shifts1[c2]
            shifts[2] = self.shifts1[c2]
            shifts[3] = self.shifts1[c2]
        else:
            shifts[0] = self.shifts1[c2+1]
            shifts[1] = self.shifts1[c2]
            shifts[2] = self.shifts1[c2+1]
            shifts[3] = self.shifts1[c2]
        return shifts.astype(int) 

    def idx_shift2(self, count):
        shifts = np.zeros(4)
        c2 = count // 2
        if count % 2 == 0:
            shifts[0] = self.shifts2[c2]
            shifts[1] = self.shifts2[c2]
            shifts[2] = self.shifts2[c2]
            shifts[3] = self.shifts2[c2]
        else:
            shifts[0] = self.shifts2[c2+1]
            shifts[1] = self.shifts2[c2+1]
            shifts[2] = self.shifts2[c2]
            shifts[3] = self.shifts2[c2]
        return shifts.astype(int)
    
    def _calculate_patch_order(self, count1, count2):
        """Determines the appropriate patch order based on shift indices."""
        patch_order = [0, 1, 2, 3]
        if count1 % 2 == 1:
            # reorder to [2, 3, 0, 1]
            patch_order = [patch_order[1], patch_order[0], patch_order[3], patch_order[2]]
        if count2 % 2 == 1:
            # reorder to [1, 0, 3, 2]
            patch_order = [patch_order[2], patch_order[3], patch_order[0], patch_order[1]]
        return np.array(patch_order).astype(int)
    
    def _prepare_shift_matrix(self):
        """Creates a matrix of shift combinations for all valid shifts."""
        shift_matrix = np.zeros((self.max_patch, self.max_shift, self.max_shift, 4))
        for k in range(self.max_patch):
            for i in range(self.max_shift):
                for j in range(self.max_shift):
                    shift_matrix[k][i][j] = (self.idx_shift1(i) + self.idx_shift2(j)+ (np.arange(4) * self.base_pix_per_chunk + k*self.nside**2))[self._calculate_patch_order(i,j)]
        return shift_matrix.astype(int)

    def get_shifted_mapchunks(self, shifted_indices, hpmap):
        """Extracts shifted map chunks from a HEALPix map."""
        return [ hpmap[shifted_indices[i] : shifted_indices[i] + self.base_pix_per_chunk] for i in range(4)]

    def bring2original(self, shifted_indices, original, map_chunks):
        """Updates the original map with shifted chunks."""
        for i in range(4):
            original[shifted_indices[i] : shifted_indices[i] + self.base_pix_per_chunk] = map_chunks[i]
