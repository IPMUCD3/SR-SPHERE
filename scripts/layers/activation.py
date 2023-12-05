import torch.nn as nn  

'''
This code is the choice of activation function.
'''

# The Acts class inherits from torch.nn.Module.
# This class is used to apply different activation functions.
class Acts(nn.Module):
    def __init__(self, act_type):
        # Initialize the constructor of the superclass
        super().__init__()

        # Select different activation functions based on the act_type parameter
        if act_type == "mish":
            self.act = nn.Mish()  # Mish activation function
        elif act_type == "silu":
            self.act = nn.SiLU()  # SiLU activation function (formerly known as Swish)
        elif act_type == "lrelu":
            self.act = nn.LeakyReLU(0.1)  # LeakyReLU activation function
        else:
            self.act = nn.ReLU()  # Default to ReLU activation function

    # The forward method applies the activation function to the input data
    def forward(self, x):
        return self.act(x)  # Return the result after applying the activation function

