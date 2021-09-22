from typing import Tuple
import torch
from torch import nn

class AudacityModel(nn.Module):

  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model

  @torch.jit.ignore
  def validate_metadata(self, metadata: dict):
    # TODO: download the schema file from audacity and check w it
    raise NotImplementedError()

  @staticmethod
  @torch.jit.ignore
  def get_example_inputs():
    """ 
    returns a list of waveform audio tensors for testing,
    shape (n_channels, n_samples). 
    """ 
    return [torch.randn(n, s) for (n, s) in zip([1 for _ in range(5)], 
                                            [3200, 3498, 48000, 32000, 88000])]
  
class WaveformToWaveform(AudacityModel):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class WaveformToLabels(AudacityModel):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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
 
class AsteroidWrapper(WaveformToWaveform):

  def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.separate(x)[0]