from pathlib import Path
import random
from typing import Tuple
from audacitorch import AudacityModel

import torch
import json

def get_example_inputs(multichannel: bool = False):
  """
  returns a list of possible input tensors for an AudacityModel. 

  Possible inputs are audio tensors with shape (n_channels, n_samples). 
  If multichannel == False, n_channels will always be 1. 
  """
  max_channels = 2 if multichannel else 1
  num_inputs = 10
  channels = [random.randint(1, max_channels) for _ in range(num_inputs)]
  sizes = [random.randint(2048, 396000) for _ in range(num_inputs)]
  return [
    torch.randn((c, s)) for c, s in  zip(channels, sizes)
  ]


