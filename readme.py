import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class MyVolumeModel(nn.Module):

    def forward(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        # do the neural net magic!
        x = x * gain

        return x

from audacitorch import ModelCard, WaveformToWaveformBase, ContinuousCtrl

class MyVolumeModelWrapper(WaveformToWaveformBase):
    
    def do_forward_pass(self, 
            x: torch.Tensor, 
            params: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        # do any preprocessing here! 
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        # we assume the user has modified the gain at this point
        output = self.model(
            x, gain=torch.tensor(params['gain'])
        )

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
    
        return output

# sample rate could be negative? 
model_card = ModelCard(
    name="MyVolumeModel",
    author="Me",
    description="A simple volume model",
    tags=["volume", "gain", "amplify"],
    sample_rate=44100
)

from pathlib import Path
from audacitorch.utils import get_example_inputs

# create a root dir for our model
root = Path('.')
root.mkdir(exist_ok=True, parents=True)

# get our model
model = MyVolumeModel()
print(f"created model: {model}")

ctrls = {
    "gain" : ContinuousCtrl(
        name="gain",
        default=1.0,
        min=0.0,
        max=10.0, 
    )
}

# wrap the model in the TensorJuceModel wrapper, which will handle all the metadata and jit.scripting
serialized_model = torch.jit.script(MyVolumeModelWrapper(model, model_card, ctrls))
print(f"serialized model: {serialized_model}")

# take your model for a test run!
sample_rate_in = 16000
audio = get_example_inputs(multichannel=False)[0]
print(f"input audio: {audio}")

# resample the audio to the model's sample rate
audio = serialized_model.resample(audio, sample_rate_in)


output = serialized_model(audio, {"gain": torch.tensor(3.0)})
print(f"output audio: {output}")

# save!
torch.jit.save(serialized_model, root / 'volumizer.pt')

# load!
loaded_model = torch.jit.load(root / 'volumizer.pt')

print(f"loaded a model {loaded_model}")
print(f"model card: {loaded_model.model_card}")
print(f"ctrls: {loaded_model.ctrls}")

breakpoint()
loaded_output = loaded_model(audio, {"gain": torch.tensor(3.0)})
assert torch.allclose(output, loaded_output)