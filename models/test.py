import torch
from torch import nn
import torch.nn.functional as F

# CNN output size
def calculate_Conv2d_size(n_in, kernel_size, stride=1, padding=0, dilation=1):
    n_out = (n_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(n_out)

def calculate_ConvTranspose2d_size(n_in, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    n_out = (n_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return int(n_out)


if __name__ == '__main__':
    # print(calculate_Conv2d_size(32, 3, 2, 1))
    print(calculate_ConvTranspose2d_size(1, kernel_size=4, stride=1, dilation=1))
    print(calculate_ConvTranspose2d_size(4, kernel_size=5, stride=1, dilation=1))
    print(calculate_ConvTranspose2d_size(8, kernel_size=4, stride=2, padding=1, dilation=1))
    print(calculate_ConvTranspose2d_size(16, kernel_size=4, stride=2, padding=1, dilation=1))
    print(calculate_ConvTranspose2d_size(32, kernel_size=4, stride=2, padding=1, dilation=1))
    # print(calculate_ConvTranspose2d_size(32, kernel_size=4, stride=2, padding=1, dilation=1))
    # print(calculate_ConvTranspose2d_size(16, kernel_size=9, stride=1, dilation=2))
    # print(calculate_ConvTranspose2d_size(55, kernel_size=5, stride=1, dilation=1))
    # print(calculate_ConvTranspose2d_size(59, kernel_size=6, stride=1, dilation=1))
    # print(calculate_Conv2d_size(32, 1, 2, 0))