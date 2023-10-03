def get_params(verbose=False):
    """Parameters for the models"""

    params = dict()
    
    #data info
    params["hrmaps_dir"]= "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/HR/z00/"
    params["lrmaps_dir"]="/gpfs02/work/akira.tokiwa/gpgpu/FastPM/LR/z00/"
    params['nside_hr'] = 64
    params['nside_lr'] = 64
    
    params['rate_train'] = 0.8
    params['n_maps'] = 100
    params['N_train'] = int(params['n_maps']*params['rate_train'])
    params['N_val'] = int(params['n_maps'] - params['N_train'])
    
    #spliting data
    params['order'] = 2
    params['issplit'] = True

    #normalization
    params['normalize'] = False
    
    #Architecture
    params['kernel_size'] = 30
    
    # Training.
    params['num_epochs'] = 1000  # Number of passes through the training data.
    params['batch_size'] = 1   # Constant quantity of information (#pixels) per step (invariant to sample size).
    params['lr_init'] = 1*10**-5
    params['lr_max'] = 1*10**-2
    
    if params['issplit']:
        params['N_train'] = int(params['n_maps']*(12 * params['order']**2)*params['rate_train'])
        params['N_val'] = int(params['n_maps']*(12 * params['order']**2) - params['N_train'])
        params['batch_size'] = 128

    params['steps_per_epoch'] = params['N_train'] // params['batch_size']
    
    if verbose:
        print('#LRsides: {0}, HRsides: {1} '.format(params['nside_lr'], params['nside_hr']))
        print('#LRpixels: {0} / {1} = {2}'.format(12*params['nside_lr']**2, 12*params['order']**2, (params['nside_lr']//params['order'])**2))
        # Number of pixels on the full sphere: 12 * nside**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*(params['nside_lr']//params['order'])**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*params['N_train']*(params['nside_lr']//params['order'])**2))
        print('Learning rate will start at {0:.1e} and reach maximum at {1:.1e}.'.format(params['lr_init'], params['lr_max']))

    return params