
# Spherical Super-Resolution in cosmological simulations
## 1. Introduction

N-body simulations are a powerful tool for studying the evolution of the universe.
The high-order statistics like non-Gaussianity only become apparent in small scales which need high resolution.
However, the computational cost of these simulations limits the resolution of the data.
At the same time, we need numbers of simulations to reduce the statistical error.
Therefore, we need to obtain high-resolution data more efficiently.

Astronomical observations offer unique challenges, with data often projected onto the celestial sky sphere. 
Traditional planar methods falter in this context, introducing distortions, especially near poles or when merging vast sky surveys. 
As computational methods advanced, we now have the tools to properly address these challenges and represent these observations in their natural form.

Our project aims to enhance the resolution of spherical data representations, particularly in astronomical observations.
Instead of directly calculate the high-resolution data, we use a generative model and the low-resolution data to generate the high-resolution data.
This method is supposed to be more efficient than directly calculate the high-resolution data.

## 2. Methodology
Our approach leverages the strengths of two models:

DEEPSPHERE (https://arxiv.org/abs/1810.12186): A variant of Spherical CNN renowned for its rotational equivariance and efficiency. However, it's traditionally utilized for classification tasks.
Diffusion Model (DDPM, https://arxiv.org/abs/2006.11239): A generative model celebrated for its stability, the diversity of generation and ease of training. 
Despite the generation time being slow, it's still faster than the simulation time of the N-body simulation.
The integration of these models ensures accuracy, computational efficiency, and the retention of DEEPSPHERE's rotational equivariance.

We utilize the structure of SR3 ( https://arxiv.org/pdf/2104.07636.pdf ) to train the model.
The planar convolutions in the denoising Unet are replaced by DEEPSPHERE.
For other parts, we use the same structure as SR3 including skip connections and self-attention.

## 3. Data
We use the data from the N-body simulation: FastPM (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.00476)
The data is simulated in a box with the size of 384 Mpc/h.
The particle number is 128^3 for the low-resolution data and 256^3 for the high-resolution data.
Then we chopped out a ball with the radius of 128 Mpc/h from the center of the box and project it onto the celestial sphere in HEALPix format.
The resolution of the data are both Nside=512.
For memory reason, we divide the data into 192 patches according to the HEALPix pixelization.

## 4. Results
Our spherical super-resolution outperforms traditional methods, especially on the celestial sphere's data. 
Key metrics include:
![Orthogonal HEALPix Map through Diffusion](https://github.com/IPMUCD3/SR-SPHERE/assets/26876924/e7d1ce1a-e267-4459-922b-fa9396b9a27c)


![Power Spectrum thorough Diffusion](https://github.com/IPMUCD3/SR-SPHERE/assets/26876924/d7da3e2e-24a3-49f8-9873-db7486c060a6)

## 5. Future Directions and Discussion
We aim to transition from simulations to real-world observations, particularly targeting platforms like LSST and Simon's observatory. Real-world data introduces complexities like noise and irregularities, challenging our model's robustness. As we refine our approach, we remain committed to ensuring the highest accuracy in high-resolution results.
