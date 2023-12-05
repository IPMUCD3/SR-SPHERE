
import torch
import numpy as np
import pytorch_lightning as pl

def get_sigmas(sigma_min, sigma_max, num_scales):
    sigmas = np.exp(
        np.linspace(np.log(sigma_max), np.log(sigma_min), num_scales))

    return sigmas

class VESDE():
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq. 2 in Song et al. 2021
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                    self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
    
    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()
    
class ReverseDiffusionPredictor():
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None] * z
        return x, x_mean

class LangevinCorrector():
    def __init__(self, sde, score_fn, snr, n_steps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise

        return x, x_mean
    
def get_loss_fn(sde, continuous=True, likelihood_weighting=True):
    def loss_fn(model, x, t, condition=None):
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_data = mean + std[:, None, None] * z
        if continuous:
            time=sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            time = sde.T - t
        time *= sde.N - 1
        time = torch.round(time).long()
        score = model(perturbed_data, time, condition=condition) if condition is not None else model(perturbed_data, time)
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None] + z)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None])
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        loss = torch.mean(losses)
        return loss
    return loss_fn

class DSM(pl.LightningModule):
    def __init__(self, model, params, sde=None):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = model(params)
        self.sde = sde
        self.continuous = params['diffusion']['continuous']
        self.likelihood_weighting = params['diffusion']['likelihood_weighting']
        self.eps = params['diffusion']['eps']
        self.loss_fn = get_loss_fn(self.sde, continuous=True, likelihood_weighting=True)
        self.batch_size = params["train"]['batch_size']
        self.learning_rate = params["train"]['learning_rate']
        self.gamma = params["train"]['gamma']
        print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))
        
    def training_step(self, batch, batch_idx):
        x, cond = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.eps) + self.eps
        loss = self.loss_fn(self.model, x, t, condition=cond) if cond is not None else self.loss_fn(self.model, x, t)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cond = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.eps) + self.eps
        loss = self.loss_fn(self.model, x, t, condition=cond) if cond is not None else self.loss_fn(self.model, x, t)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 
