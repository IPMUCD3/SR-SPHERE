
def hp_split(img, order, nest=True):
    """
    Function to split the image into multiple images based on the given order.
    """
    npix = len(img)
    nsample = 12 * order**2
    
    if npix < nsample:
        raise ValueError('Order not compatible with data.')
    
    if not nest:
        raise NotImplementedError('Implement the change of coordinate.')
    
    return img.reshape([nsample, npix // nsample])