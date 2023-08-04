from typing import Tuple, List, Optional, Type, Any, Dict, Union
import torch
from torch import nn
from torchaudio.functional import resample
from dataclasses import dataclass

from .utils import get_list_type, get_dict_types

def _waveform_check(x: torch.Tensor):
  assert x.ndim == 2, "input must have two dimensions (channels, samples)"
  assert x.shape[-1] > x.shape[0], f"The number of channels {x.shape[-2]} exceeds the number of samples {x.shape[-1]} in your INPUT waveform. \
                                      There might be something wrong with your model. "

@torch.jit.script
@dataclass
class ContinuousCtrl:
  name: str
  min: float
  max: float
  default: float

@torch.jit.script
@dataclass
class ChoiceCtrl:
  name: str
  choices: List[str]
  default: str

# define generic control type
Ctrl = Union[ContinuousCtrl, ChoiceCtrl]

@torch.jit.script
@dataclass
class ModelCard:
  sample_rate: int
  name: str
  author: str
  description: str
  tags: List[str]


class TensorJuceModel(nn.Module):
  model_card: ModelCard
  ctrls: Dict[str, Ctrl]

  def __init__(self, 
      model: nn.Module, 
      model_card: ModelCard, # essential metadata about the model
      ctrls: Dict[str, Ctrl] # a dictionary of parameters that can be controlled by the user
    ):
    """ creates an Audacity model, wrapping a child model (that does the real work)"""
    super().__init__()
    model.eval()
    self.model = model

    attributes = {}
    attributes['model_card'] = model_card
    attributes['ctrls'] = ctrls
    self.register_attributes(attributes)

  @torch.jit.ignore
  def register_attributes(self, attributes: Dict[str, Any]):
    """registers attributes on the model"""
    for attr_key, attr_val in attributes.items():
      attr_type = type(attr_val)

      if attr_key == 'ctrls':
        attr_type = Dict[str, Ctrl] # torchscript doesn't support inheritance!
      elif attr_type == list:
        attr_type = get_list_type(attr_val)
      elif attr_type == dict:
        attr_type = get_dict_types(attr_val)

      attr_val = torch.jit.Attribute(attr_val, attr_type)
      print(f"registering attribute {attr_key} of type {attr_type}")
      setattr(self, attr_key, attr_val) 

class WaveformToWaveformBase(TensorJuceModel):

  def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """ 
    Internal forward pass for a WaveformToWaveform model. 

    All this does is wrap the do_forward_pass(x) function in assertions that check 
    that the correct input/output constraints are getting met. Nothing fancy. 
    """
    _waveform_check(x)
    x = self.do_forward_pass(x, params)
    _waveform_check(x)
    
    return x

  def do_forward_pass(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """ 
    Perform a forward pass on a waveform-to-waveform model.
    
    Args:
        x : An input audio waveform tensor. If `"multichannel" == True` in the 
            model's `metadata.json`, then this tensor will always be shape 
            `(1, n_samples)`, as all incoming audio will be downmixed first. 
            Otherwise, expect `x` to be a multichannel waveform tensor with 
            shape `(n_channels, n_samples)`.
        params : A tensor of UI parameters to use for the forward pass. Optional. 

    Returns:
        torch.Tensor: Output tensor, shape (n_sources, n_samples). Each source 
                      will be  
    """
    raise NotImplementedError("implement me!")

  @torch.jit.export
  def resample(self, x: torch.Tensor, sample_rate_in: int):
    return resample(x, sample_rate_in, self.model_card.sample_rate)
