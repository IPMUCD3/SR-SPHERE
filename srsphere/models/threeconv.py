from srsphere.models.model_template import template

from srsphere.utils.partial_laplacians import get_partial_laplacians
from srsphere.utils.cheby_shev import SphericalChebConv
from torch import nn

class test_twoconv(template):
    def __init__(self, params):
        super().__init__(params)
        self.kernel_size= params["kernel_size"]
        self.laps = get_partial_laplacians(params['nside_lr'], 1, params['order'], 'normalized')
        self.conv_in = SphericalChebConv(1, 64, self.laps[0], self.kernel_size)
        self.conv_mid = SphericalChebConv(64, 64, self.laps[0], self.kernel_size)
        self.conv_out = SphericalChebConv(64, 1, self.laps[0], self.kernel_size)
        self.ReLU = nn.PReLU()
    
    def forward(self, x):
        x = self.ReLU(self.conv_in(x))
        x = self.ReLU(self.conv_mid(x))
        x = self.conv_out(x)
        return x