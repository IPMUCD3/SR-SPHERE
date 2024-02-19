
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srsphere.diffusion.scheduler import cosine_beta_schedule, linear_beta_schedule
from srsphere.diffusion.utils import extract

class Diffusion():
    def __init__(self, **args):
        self.timesteps = args['timesteps']
        if args['schedule'] == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps, s=args['cosine_beta_s'])
            print("The schedule is cosine with s = {}".format(args['cosine_beta_s']))
        elif args['schedule'] == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=args['linear_beta_start'], beta_end=args['linear_beta_end'])
            print("The schedule is linear with beta_start = {}, beta_end = {}".format(args['linear_beta_start'], args['linear_beta_end']))

        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # y_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        #self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas
        # y_t-1 = sqrt_recip_alphas * (y_t - betas_t * MODEL(x_t, t) / sqrt_one_minus_alphas_cumprod_t)
        #         + sqrt_one_minus_alphas_cumprod_t * eps_t

        self.loss_type = args["loss_type"]
        if self.loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif self.loss_type == 'l2':
            self.loss_fn = F.mse_loss
        elif self.loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, condition=None):
        # L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)
        loss = self.loss_fn(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def timewise_loss(self, denoise_model, x_start, t, noise=None, condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)
        loss = self.loss_fn(noise, predicted_noise)
        loss = torch.mean(loss, dim=[-2, -1]) #mean over all spatial dims
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, condition=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_output = model(x, t) if condition is None else model(x, t, condition=condition)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + posterior_variance_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition=None):
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        if condition is not None:
            assert condition.shape[0] == shape[0]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition=condition)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1, condition=None):
        return self.p_sample_loop(model, shape=(batch_size, image_size, channels), condition=condition)