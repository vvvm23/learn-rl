import torch

from typing import Tuple
def get_conv_out(layer: torch.nn.Module, in_shape: Tuple[int, int, int]):
    out_shape = layer(torch.zeros(1, *in_shape)).shape
    return out_shape[1]*out_shape[2]*out_shape[3]
