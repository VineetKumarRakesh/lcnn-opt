"""
Custom layer definitions and reusable blocks for model architectures.
Add any specialized layers here as needed (e.g., custom blocks for new architectures).
Currently serves as a placeholder for future extensions.
"""

# Example: simple Conv-BatchNorm-Activation block
import torch.nn as nn

class ConvBNAct(nn.Sequential):
    """
    A convenience block: Conv2d -> BatchNorm2d -> Activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Convolution kernel size.
        stride (int or tuple): Convolution stride.
        padding (int or tuple): Convolution padding.
        groups (int): Number of convolution groups.
        act_layer (nn.Module): Activation layer class (default: nn.ReLU).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 act_layer=nn.ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(inplace=True)
        )
