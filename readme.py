import torch
import torch.nn as nn
from typing import Optional 

class MyVolumeModel(nn.Module):

    def forward(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        # do the neural net magic!
        x = x * gain

        return x

from audacitorch import WaveformToWaveformBase

class MyVolumeModelWrapper(WaveformToWaveformBase):
    
    def do_forward_pass(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # do any preprocessing here! 
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        output = self.model(x, params)

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
    
        return output

metadata = {
    'name': 'Volume Booster',
    'author': 'Hugo Flores Garcia',
    'sample_rate': 48000, 
    'domain_tags': ['music', 'speech', 'environmental'],
    'description': 'Use me to boost volume by 3dB.',
    'tags': ['volume boost'],
    'labels': ['boosted'],
    'effect_type': 'waveform-to-waveform',
    'multichannel': False,
}

from pathlib import Path
from audacitorch.utils import get_example_inputs

# create a root dir for our model
root = Path('booster-net')
root.mkdir(exist_ok=True, parents=True)

# get our model
model = MyVolumeModel()
print(f"created model: {model}")

# wrap the model in the TensorJuceModel wrapper, which will handle all the metadata and jit.scripting
serialized_model = MyVolumeModelWrapper(model, metadata)
print(f"serialized model: {serialized_model}")

# take your model for a test run!
audio = get_example_inputs(multichannel=False)[0]
print(f"input audio: {audio}")

gain = torch.tensor(3.0)
output = serialized_model(audio, gain)
print(f"output audio: {output}")

# save!
torch.jit.save(serialized_model, root / 'volumizer.pt')