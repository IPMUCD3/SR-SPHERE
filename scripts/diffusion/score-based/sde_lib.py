"""Copied from https://github.com/GBATZOLIS/conditional_score_diffusion/tree/true_master"""

"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import torch
import numpy as np    

class VESDE():
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, data_mean=None):
        """Construct a Variance Exploding SDE.
        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

        self.diffused_mean = data_mean #new

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))).type_as(t))
        return drift, diffusion

    def marginal_prob(self, x, t): #perturbation kernel P(X(t)|X(0)) parameters
        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = sigma_min * (sigma_max / sigma_min) ** t
        mean = x
        return mean, std
    
    def compute_backward_kernel(self, x0, x_tplustau, t, tau):
        #x_forward = x(t+\tau)
        #compute the parameters of p(x(t)|x(0), x(t+\tau)) - the reverse kernel of width tau at time step t.
        sigma_min, sigma_max = torch.tensor(self.sigma_min).type_as(t), torch.tensor(self.sigma_max).type_as(t)

        sigma_t_square = (sigma_min * (sigma_max / sigma_min) ** t)**2
        sigma_tplustau_square = (sigma_min * (sigma_max / sigma_min) ** (t+tau))**2

        std_backward = torch.sqrt(sigma_t_square * (sigma_tplustau_square - sigma_t_square) / sigma_tplustau_square)

        #backward scaling factor for the mean
        s_b_0 = (sigma_tplustau_square - sigma_t_square) / sigma_tplustau_square
        s_b_tplustau = sigma_t_square / sigma_tplustau_square

        mean_backward = x0 * s_b_0[(...,) + (None,) * len(x0.shape[1:])] + x_tplustau * s_b_tplustau[(...,) + (None,) * len(x0.shape[1:])]

        return mean_backward, std_backward

    def prior_sampling(self, shape):
        if self.diffused_mean is not None:
            repeat_tuple = tuple([shape[0]]+[1 for _ in shape[1:]])
            diffused_mean = self.diffused_mean.unsqueeze(0).repeat(repeat_tuple)
            return torch.randn(*shape) * self.sigma_max + diffused_mean
        else:
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
                drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        return RSDE()


class cVESDE():
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, data_mean=None):
        """Construct a Variance Exploding SDE.
        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.diffused_mean = data_mean #new

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t): #perturbation kernel P(X(t)|X(0)) parameters 
        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = sigma_min * (sigma_max / sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        if self.diffused_mean is not None:
            repeat_tuple = tuple([shape[0]]+[1 for _ in shape[1:]])
            diffused_mean = self.diffused_mean.unsqueeze(0).repeat(repeat_tuple)
            return torch.randn(*shape) * self.sigma_max + diffused_mean
        else:
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

            def sde(self, x, y, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score_x = score_fn(x, y, t) #conditional score on y
                drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_x * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, y, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_fn(x, y, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        return RSDE()