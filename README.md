# torchaudacity

This package contains utilities for prepping PyTorch audio models for use in Audacity. More specifically, it provides abstract classes for you to wrap your waveform-to-waveform and waveform-to-labels models (see the [Deep Learning for Audacity]() **TODO: add link** website to learn what those are).   

## Usage

`torchaudacity` contains two abstract classes for serializing two types of models: waveform-to-waveform and waveform-to-labels. The classes are `WaveformToWaveform`, and `WaveformToLabels`. 

### Waveform to Waveform

Waveform-to-waveform models receive a single multichannel audio track as input, and may write to a variable number of new audio tracks as output.

### Waveform to Labels

Waveform-to-labels models receive a single multichannel audio track as input, and may write to an output label track as output. 

### Example - Waveform-to-Waveform

Here's a minimal example for a model that simply boosts volume by multiplying the incoming audio by a factor of 2. 

#### Define a model class

First, we create our model. There are no internal constraints on what the, as long as you can use `torch.jit.script` or `torch.jit.trace`, and it is able to meet the input-output constraints specified in waveform-to-waveform and waveform-to-labels models. 

```python
import torch.nn as nn

class MyVolumeModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do the neural net magic!
        x = x * 2

        return x
```

#### Create a wrapper class

Now, we create a wrapper class for our model. Because our model returns an audio waveform as output, we'll use `WaveformToWaveform` as our parent class. For both `WaveformToWaveform` and `WaveformToLabels`, we need to implement the `do_forward_pass` method with our processing code. See the [docstrings](/torchaudacity/core.py) for more details. 

```python
from torchaudacity import WaveformToWaveform

class MyVolumeModelWrapper(WaveformToWaveform):

    def __init__(self, model):
        model.eval()
        self.model = model
    
    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        
        # do any preprocessing here! 
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        output = self.model(x)

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
    
        return output
```

#### Create a metadata file

Audacity models need a metadata file. See the metadata [schema: TODO NEEDS LINK]() to learn about the required fields. 

```python
metadata = {
    'sample_rate': 48000, 
    'domain_tags': ['music', 'speech', 'environmental'],
    'short_description': 'Use me to boost volume by 3dB :).',
    'long_description':  'This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.',
    'tags': ['volume boost'],
    'labels': ['boosted'],
    'effect_type': 'waveform-to-waveform',
    'multichannel': False,
}

```

All set! We can now proceed to serialize the model to torchscript and save the model, along with its metadata.

```python
from pathlib import Path
from torchaudacity import save_model

# create a root dir for our model
root = Path('booster-net')
root.mkdir(exist_ok=True, parents=True)

# get our model
model = MyVolumeModel()

# wrap it
wrapper = MyVolumeModelWrapper(model)

# serialize it
# an alternative is to use torch.jit.trace
serialized_model = torch.jit.script(serialized_model)

# save!
save_model(serialized_model, metadata, root)
```

#### Export to HuggingFace

You should now have a directory structure that looks like this: 

```
/booster-net/
/booster-net/model.pt
/booster-net/metadata.json
```

This will be the repository for your audacity model. Make sure to add a readme with the `audacity` tag in the YAML metadata, so it show up on the explore tab of Audacity's Deep Learning Tools. 

Create a `README.md` inside `booster-net/`, and add the following header:


in README.md
```
---
tags: audacity
---
```

Awesome! It's time to push to HuggingFace. See their [documentation](https://huggingface.co/docs/hub/adding-a-model) for adding a model to the HuggingFace model hub. 