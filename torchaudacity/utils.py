from pathlib import Path
import random

import torch
import json

def save_model(model: torch.jit.ScriptModule, metadata: dict, root_dir: Path):
  """
  Save a compiled torch.jit.ScriptModule, along with a metadata dictionary.

  Args:
      model: your Audacity-ready serialized model, using either torch.jit.trace or torch.jit.script. 
        Should derive from torchaudacity.WaveformToWaveformBase or torchaudacity.WaveformToLabelsBase.
      metadata: a metadata dictionary. Shoule be validated using torchaudio.utils.validate_metadata()

  Returns:
    Will create the following files: 
    ```
      root_dir/
      root_dir/model.pt
      root_dir/metadata.json
    ```
  """
  root_dir.mkdir(exist_ok=True, parents=True)

  # save model and metadata!
  torch.jit.save(model, root_dir / 'model.pt')

  with open(root_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f)


def validate_metadata(metadata: dict) -> bool:
  raise NotImplementedError

def get_example_inputs(multichannel: bool = False):
  """
  returns a list of possible input tensors for an AudacityModel. 

  Possible inputs are audio tensors with shape (n_channels, n_samples). 
  If multichannel == False, n_channels will always be 1. 
  """
  max_channels = 10 if multichannel else 1
  num_inputs = 10
  channels = [random.randint(1, max_channels) for _ in range(num_inputs)]
  sizes = [random.randint(2048, 396000) for _ in range(num_inputs)]
  return [
    torch.randn((random.randint(1, 4)))
  ]
