
from calendar import c
import numpy as np
import healpy as hp
import torch
from tqdm import tqdm

from srsphere.generation.hp_shifting import NestGridShift

class GenerationStrategy:
    """
    Implements a map generation strategy using a Nested Grid Shift and a generation function.
    """

    def __init__(self, model: torch.nn.Module, nside: int, order: str, params: dict, cond=None):
        """
        Initializes the GenerationStrategy object.

        Args:
            model: The model used for map generation.
            nside: HEALPix resolution parameter (NSIDE).
            order: The resolution parameter (NSIDE) of a patch.
            params: Dictionary of parameters for the map generation process.
            cond: Optional initial condition map. 
        """

        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.order = order
        self.grid = NestGridShift(nside, order)

        self.model = model
        self.device = model.device
        self.timesteps = int(params["timesteps"])
        self.save_dir = params["save_dir"]

        self.img = np.zeros(self.npix)
        self.gen_flag = np.zeros([self.grid.max_shift + 1, self.grid.max_shift + 1]).astype(int)
        self.cond = cond

    def _extract_mask(self, count1: int, count2: int) -> list:
        """Extracts the generation mask for adjacent superpixels."""
        return np.array([
            self.gen_flag[count1, count2],
            self.gen_flag[count1 + 1, count2],
            self.gen_flag[count1, count2 + 1],
            self.gen_flag[count1 + 1, count2 + 1]
        ])[self.grid._calculate_patch_order(count1, count2)]

    def _fill_gen_flag(self, count1: int, count2: int) -> None:
        """Updates the generation flags for the current and adjacent superpixels."""
        for i in range(count1, count1 + 2):
            for j in range(count2, count2 + 2):
                self.gen_flag[i, j] = 1

    def _prepare_data_tensors(self, count1: int, count2: int) -> tuple:
        """Prepares condition and image segment tensors for a superpixel."""
        idx_chunks = self.grid.shift_matrix[:, count1, count2]
        conds = [np.hstack(self.grid.get_shifted_mapchunks(idx, self.cond)) for idx in idx_chunks]
        imgs = [np.hstack(self.grid.get_shifted_mapchunks(idx, self.img)) for idx in idx_chunks]
        return (
            torch.from_numpy(np.array(conds)).unsqueeze(2).float().to(self.device),
            torch.from_numpy(np.array(imgs)).unsqueeze(2).float().to(self.device)
        )
    
    def _generate_from_model(self, conds: torch.Tensor, imgs: torch.Tensor, mask: list) -> torch.Tensor:
        """Generates map segments for a superpixel using the model.

        Args:
            conds: Condition map segments.
            imgs: Image map segments.
            mask: Generation mask for the superpixel.
        
        Returns:
            torch.Tensor: Generated map segments.
        """
        imgs = torch.chunk(imgs, 4, dim=1)
        y = torch.randn(conds.shape, device=self.device)
        with torch.no_grad():
            for j in tqdm(reversed(range(0, self.timesteps)), desc="Diffusion", total=self.timesteps):
                t = torch.full((y.shape[0],), j, device=y.device, dtype=torch.long)
                # replace masked chunk with pre-generated map segment
                y = torch.chunk(y, 4, dim=1)
                y = torch.cat(tuple([y[i] * (1 - mask[i]) + imgs[i] * mask[i] for i in range(4)]), dim=1)
                y = self.model.diffusion.p_sample(self.model.model, y, t, t_index=j, condition=conds)
        return y

    def _generate_and_update(self, count1: int, count2: int, save=True) -> None:
        """Generates map segments for a superpixel and updates the main map."""
        conds, imgs = self._prepare_data_tensors(count1, count2)
        mask = self._extract_mask(count1, count2)
        generated = self._generate_from_model(conds, imgs, mask).squeeze(2).detach().cpu().numpy()

        for patch_num, chunk in enumerate(generated):
            idx = self.grid.shift_matrix[patch_num, count1, count2]
            self.img[idx[0]:idx[0] + self.grid.base_pix_per_chunk] = chunk[0:self.grid.base_pix_per_chunk] 
            self.img[idx[1]:idx[1] + self.grid.base_pix_per_chunk] = chunk[self.grid.base_pix_per_chunk:2*self.grid.base_pix_per_chunk] 
            self.img[idx[2]:idx[2] + self.grid.base_pix_per_chunk] = chunk[2*self.grid.base_pix_per_chunk:3*self.grid.base_pix_per_chunk]
            self.img[idx[3]:idx[3] + self.grid.base_pix_per_chunk] = chunk[3*self.grid.base_pix_per_chunk:4*self.grid.base_pix_per_chunk]
        self._fill_gen_flag(count1, count2) 
        if save:
            np.save(f"{self.save_dir}/step_{str(count1).zfill(3)}_{str(count2).zfill(3)}.npy", self.img)

    def generation_task(self) -> None:
        """Performs the complete map generation process."""
        # generate the center of the map
        self._generate_and_update(0, 0)
        print("Generated the first cell of the map.")
        # generate the edge of the map
        for count1 in range(self.grid.max_shift):
            self._generate_and_update(count1, 0)
        print("Generated the first column of the map.")
        for count2 in range(self.grid.max_shift):
            self._generate_and_update(0, count2)
        print("Generated the first row of the map.")
        # generate the body of the map
        for count1 in range(1, self.grid.max_shift):
            for count2 in range(1, self.grid.max_shift):
                self._generate_and_update(count1, count2)
        print("Generated the body of the map.")