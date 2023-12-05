


def get_inverse_problem_smld_loss_fn(sde, train, reduce_mean=False, likelihood_weighting=True):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    # Previous SMLD models assume descending sigmas
    smld_sigma_array_y = sde['y'].discrete_sigmas #observed
    smld_sigma_array_x = sde['x'].discrete_sigmas #unobserved
    

    def loss_fn(model, batch):
        y, x = batch
        score_fn = mutils.get_score_fn(sde, model, train=train)
        labels = torch.randint(0, sde['x'].N, (x.shape[0],), device=x.device)
        score_fn_labels = labels/(sde['x'].N - 1)

        sigmas_y = smld_sigma_array_y.type_as(y)[labels]
        sigmas_x = smld_sigma_array_x.type_as(x)[labels]
        
        noise_y = torch.randn_like(y) * sigmas_y[(..., ) + (None, ) * len(y.shape[1:])]
        perturbed_data_y = noise_y + y
        noise_x = torch.randn_like(x) * sigmas_x[(..., ) + (None, ) * len(x.shape[1:])]
        perturbed_data_x = noise_x + x

        perturbed_data = {'x':perturbed_data_x, 'y':perturbed_data_y}
        score = score_fn(perturbed_data, score_fn_labels)

        target_x = -noise_x / (sigmas_x ** 2)[(..., ) + (None, ) * len(x.shape[1:])]
        target_y = -noise_y / (sigmas_y ** 2)[(..., ) + (None, ) * len(y.shape[1:])]
        
        losses_x = torch.square(score['x']-target_x)
        losses_y = torch.square(score['y']-target_y)

        if likelihood_weighting:
            losses_x = losses_x*sigmas_x[(..., ) + (None, ) * len(x.shape[1:])]**2
            losses_x = losses_x.reshape(losses_x.shape[0], -1)
            losses_y = losses_y*sigmas_y[(..., ) + (None, ) * len(y.shape[1:])]**2
            losses_y = losses_y.reshape(losses_y.shape[0], -1)
            losses = torch.cat((losses_x, losses_y), dim=-1)
            losses = reduce_op(losses, dim=-1)
        else:
            losses_x = losses_x.reshape(losses_x.shape[0], -1)
            losses_y = losses_y.reshape(losses_y.shape[0], -1)
            losses = torch.cat((losses_x, losses_y), dim=-1)
            smld_weighting = (sigmas_x**2*sigmas_y**2)/(sigmas_x**2+sigmas_y**2) #smld weighting
            losses = reduce_op(losses, dim=-1) * smld_weighting
        
        loss = torch.mean(losses)
        
        return loss
    
    return loss_fn