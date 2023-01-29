from typing import Tuple
import torch
from torch import nn
from abc import abstractmethod

def _waveform_check(x: torch.Tensor):
  assert x.ndim == 2, "input must have two dimensions (channels, samples)"
  assert x.shape[-1] > x.shape[0], f"The number of channels {x.shape[-2]} exceeds the number of samples {x.shape[-1]} in your INPUT waveform. \
                                      There might be something wrong with your model. "

def _labels_check(y: Tuple):
  assert isinstance(y, tuple), "waveform-to-labels output must be a tuple"
  assert len(y) == 2, "output tuple must have two elements, e.g. tuple(labels, timestamps)"

  labels, timestamps = y

  assert torch.all(timestamps >= 0).item(), f"found a timestamp that is less than zero"

  for timestamp in timestamps:
    assert timestamp[0] < timestamp[1], f"timestamp ends ({timestamp[1]}) before it starts ({timestamp[0]})"

  assert labels.ndim == 1, "labels tensor should be one dimensional"

  assert labels.shape[0] == timestamps.shape[0], "time dimension between " \
                                                 "labels and timestamps tensors must be equal"
  assert timestamps.shape[1] == 2, "second dimension of the timestamps tensor" \
                                   "must be size 2"

# TODO - update object with actual MIDI type once decided
def _midi_check(z: object):
  # TODO
  assert True

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
    _waveform_check(x)
    x = self.do_forward_pass(x)
    _waveform_check(x)
    
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
                      will be TODO
    """
    raise NotImplementedError("implement me!")

class WaveformToLabelsBase(AudacityModel):

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Internal forward pass for a WaveformToLabels model. 

    All this does is wrap the do_forward_pass(x) function in assertions that check 
    that the correct input/output constraints are getting met. Nothing fancy. 
    """
    _waveform_check(x)
    output = self.do_forward_pass(x)
    _labels_check(output)

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

class WaveformToMidiBase(AudacityModel):

  def forward(self, x: torch.Tensor) -> object:
    """
    Internal forward pass for a WaveformToMidi model.

    All this does is wrap the do_forward_pass(x) function in assertions that check
    that the correct input/output constraints are getting met. Nothing fancy.
    """
    _waveform_check(x)
    mid = self.do_forward_pass(x)
    # TODO - use lookup table to convert to integers (tokens)
    _midi_check(mid) # TODO - or tokens check

    return mid

  def do_forward_pass(self, x: torch.Tensor) -> object:
    """
    Perform a forward pass on a waveform-to-midi model.

    Args:
        x : An input audio waveform tensor. If `"multichannel" == True` in the
            model's `metadata.json`, then this tensor will always be shape
            `(1, n_samples)`, as all incoming audio will be downmixed first.
            Otherwise, expect `x` to be a multichannel waveform tensor with
            shape `(n_channels, n_samples)`.

    Returns:
        object: TODO
    """
    raise NotImplementedError("implement me!")
