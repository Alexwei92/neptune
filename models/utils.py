import torch

# Conv2d output size
def calculate_Conv2d_size(n_in, kernel_size, stride=1, padding=0, dilation=1):
    n_out = (n_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(n_out)

# ConvTranspose2d output size
def calculate_ConvTranspose2d_size(n_in, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    n_out = (n_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return int(n_out)

# Weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)