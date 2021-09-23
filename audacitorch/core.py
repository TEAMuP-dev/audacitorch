from typing import Tuple
import torch
from torch import nn

class AudacityModel(nn.Module):

  def __init__(self, model: nn.Module):
    """ creates an Audacity model, wrapping a child model (that does the real work)"""
    super().__init__()
    model.eval()
    self.model = model

class WaveformToWaveformBase(AudacityModel):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """ 
    Internal forward pass for a WaveformToWaveform model. 

    All this does is wrap the do_forward_pass(x) function in assertions that check 
    that the correct input/output constraints are getting met. Nothing fancy. 
    """
    assert x.ndim == 2, "input must have two dimensions (channels, samples)"
    x = self.do_forward_pass(x)
    assert x.ndim == 2, "output must have two dimensions (channels, samples)"
    
    return x

  def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
    """ 
    Perform a forward pass on a waveform-to-waveform model.
    
    Args:
        x : An input audio waveform tensor. If `"multichannel" == True` in the 
            model's `metadata.json`, then this tensor will always be shape 
            `(1, n_samples)`, as all incoming audio will be downmixed first. 
            Otherwise, expect `x` to be a multichannel waveform tensor with 
            shape `(n_channels, n_samples)`.

    Returns:
        torch.Tensor: Output tensor, shape (n_sources, n_samples). Each source 
                      will be  
    """
    raise NotImplementedError("implement me!")

class WaveformToLabelsBase(AudacityModel):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """ 
    Internal forward pass for a WaveformToLabels model. 

    All this does is wrap the do_forward_pass(x) function in assertions that check 
    that the correct input/output constraints are getting met. Nothing fancy. 
    """
    assert x.ndim == 2,  "input must have two dimensions (channels, samples)"
    output = self.do_forward_pass(x)

    assert isinstance(output, tuple), "waveform-to-labels output must be a tuple"
    assert len(output) == 2, "output tuple must have two elements, e.g. tuple(labels, timestamps)"

    labels = output[0]
    timestamps = output[1]

    assert labels.shape[0] == timestamps.shape[0], "time dimension between "\
                                    "labels and timestamps tensors must be equal"
    assert timestamps.shape[1] == 2, "second dimension of the timestamps tensor"\
                                      "must be size 2"
    return output

  def do_forward_pass(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Perform a forward pass on a waveform-to-labels model.
    
    Args:
        x : An input audio waveform tensor. If `"multichannel" == True` in the 
            model's `metadata.json`, then this tensor will always be shape 
            `(1, n_samples)`, as all incoming audio will be downmixed first. 
            Otherwise, expect `x` to be a multichannel waveform tensor with 
            shape `(n_channels, n_samples)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: a tuple of tensors, where the first 
            tensor contains the output class probabilities 
            (shape `(n_timesteps, n_labels)`), and the second tensor contains 
            timestamps with start and end times for each label, 
            shape `(n_timesteps, 2)`. 

    """
    raise NotImplementedError("implement me!")
