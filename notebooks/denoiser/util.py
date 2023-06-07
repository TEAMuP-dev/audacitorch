import math
import torch

################################################################################
# Resampling utilities for DEMUCS architecture
################################################################################

@torch.jit.script
def ensure_valid_audio(x: torch.Tensor):
    """
    Convert inputs to mono (single-channel) audio if necessary
    """
    assert 1 <= x.ndim <= 3
    
    if x.ndim == 1:  # (t) --> (b, t)
        x = x.unsqueeze(0)
    if x.ndim == 2:  # (b, t) --> (b, c, t)
        x = x.unsqueeze(1)
    if x.ndim == 3:  # (b, t) --> (b, 1, t)
        x = x.mean(1, keepdim=True)
    
    return x.float()


@torch.jit.script
def sinc(x: torch.Tensor):
    """
    Sinc function.
    """
    return torch.where(
        x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(x) / x
    )


@torch.jit.script
def kernel_upsample2(zeros: int = 56):
    """
    Compute windowed sinc kernel for upsampling by a factor of 2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False, dtype=torch.float)
    win_odd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * win_odd).view(1, 1, -1)
    return kernel


@torch.jit.script
def upsample2(x: torch.Tensor, zeros: int = 56):
    """
    Upsample input by a factor of 2 using sinc interpolation.
    """
    x = ensure_valid_audio(x)
    b, c, t = x.shape
    
    kernel = kernel_upsample2(zeros).to(x)
    out = torch.nn.functional.conv1d(x.view(-1, 1, t), kernel, padding=zeros)[
        ..., 1:
    ].view(b, c, t)
    y = torch.stack([x, out], dim=-1)
    return y.view(b, c, -1)


@torch.jit.script
def kernel_downsample2(zeros: int = 56):
    """
    Compute windowed sinc kernel for downsampling by a factor of 2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False, dtype=torch.float)
    win_odd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * win_odd).view(1, 1, -1)
    return kernel


@torch.jit.script
def downsample2(x: torch.Tensor, zeros: int = 56):
    """
    Downsample input by a factor of 2 using sinc interpolation.
    """
    x = ensure_valid_audio(x)
    if x.shape[-1] % 2 != 0:
        x = torch.nn.functional.pad(x, (0, 1))
        
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    
    b, c, t = x_odd.shape
    
    kernel = kernel_downsample2(zeros).to(x)
    out = x_even + torch.nn.functional.conv1d(
        x_odd.view(-1, 1, t), kernel, padding=zeros
    )[..., :-1].view(b, c, t)
    return out.view(b, c, -1).mul(0.5)