
import torch
import yaml

def monitor_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    a = torch.cuda.memory_allocated(0)
    print("allocated memory:{0}MB, used {1:.1f}% of total".format(a//1024**2, (a/t*100)))

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        hparams = yaml.load(file, Loader=yaml.UnsafeLoader)
    return hparams

def run_reverse_test(denoising_model, diffusion, cond, start_step, device, threshold=1e1):
    imgs = []
    img = torch.randn(cond.shape, device=device)
    stats = {"mean": torch.zeros(start_step, device=device), "var": torch.zeros(start_step, device=device)}
    with torch.no_grad():
        for j in reversed(range(start_step)):
            t = torch.full((img.shape[0],), j, device=device, dtype=torch.long)
            img = diffusion.p_sample(denoising_model, img, t, t_index=j, condition=cond)
            # trace mean, var
            stats["mean"][j] = img.mean()
            stats["var"][j] = img.var()
            if j%25==0:
                print("step {}, mean: {:.3f}, var: {:.3f}".format(j, stats["mean"][j], stats["var"][j]))
                imgs.append(img.detach().cpu().numpy())
            if img.var() > threshold:
                print("Diverged at step {}, aborting.".format(j))
                return stats, imgs
    return stats, imgs