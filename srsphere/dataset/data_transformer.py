
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x / (1 - x))

class Transforms():
    """
    Class for the data transformation.

    Args:
        transform_type (str): type of the transform. Options: 'minmax', 'sigmoid', 'linear2log', 'none'.
        range_min (float): minimum value of the range for minmax transform.
        range_max (float): maximum value of the range for minmax transform.

    Attributes:
        transform_type (str): type of the transform. Options: 'minmax', 'sigmoid', 'linear2log', 'none'.
        range_min (float): minimum value of the range for minmax transform.
        range_max (float): maximum value of the range for minmax transform.
        transform (func): transform function.
        inverse_transform (func): inverse transform function.
    """
    def __init__(self, transform_type="minmax", range_min=None, range_max=None):
        self.transform_type = transform_type
        if transform_type == 'minmax':
            assert range_min is not None and range_max is not None, "range_min and range_max must be specified for minmax transform"
        self.range_min, self.range_max = range_min, range_max
        self.set_transforms()

    def set_transforms(self):
        if self.transform_type == 'minmax': # [range_min, range_max] -> [-1, 1]
            self.transform = lambda t: (t - self.range_min) / (self.range_max - self.range_min) * 2 - 1
            self.inverse_transform = lambda t: (t + 1) / 2 * (self.range_max - self.range_min) + self.range_min
        elif self.transform_type == 'sigmoid':
            self.transform = lambda t: sigmoid(t) # [-inf, inf] -> [0, 1] 
            self.inverse_transform = lambda t: logit(t)
        elif self.transform_type == 'linear2log':
            self.transform = lambda t: np.log(t + 1) # [0, inf] -> [0, inf]
            self.inverse_transform = lambda t: np.exp(t) - 1
        else: # no transform
            self.transform = lambda t:t
            self.inverse_transform = lambda t:t

