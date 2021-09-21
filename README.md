# torchaudacity

This package contains utilities for prepping PyTorch audio models for use in Audacity. More specifically, it provides abstract classes for you to wrap your waveform-to-waveform and waveform-to-labels models (see the [Deep Learning for Audacity]() **TODO: add link** website to learn more about deep learning models for audacity).   

## Usage

`torchaudacity` contains two abstract classes for serializing two types of models: waveform-to-waveform and waveform-to-labels. The classes are `WaveformToWaveform`, and `WaveformToLabels`. 

![](/torchaudacity/assets/tensor-flow.png)

### Waveform to Waveform

Waveform-to-waveform models receive a single multichannel audio track as input, and may write to a variable number of new audio tracks as output.

Example models for waveform-to-waveform effects include source separation, neural upsampling, guitar amplifier emulation, generative models, etc. Output tensors for waveform-to-waveform models must be multichannel waveform tensors with shape `(num_output_channels, num_samples)`. For every audio waveform in the output tensor, a new audio track is created in the Audacity project. 

### Waveform to Labels

Waveform-to-labels models receive a single multichannel audio track as input, and may write to an output label track as output. The waveform-to-labels effect can be used for many audio analysis applications, such as voice activity detection, sound event detection, musical instrument recognition, automatic speech recognition, etc. The output for waveform-to-labels models must be a tuple of two tensors. The first tensor corresponds to the class probabilities for each label present in the waveform, shape `(num_timesteps, num_classes)`. The second tensor must contain timestamps with start and stop times for each label, shape `(num_timesteps, 2)`.  

### Model Metadata

Certain details about the model, such as its sample rate, tool type (e.g. waveform-to-waveform or waveform-to-labels), list of labels, etc. must be provided by the model contributor in a separate `metadata.json` file. In order to help users choose the correct model for their required task, model contributors are asked to provide a short and long description of the model, the target domain of the model (e.g. speech, music, environmental, etc.), as well as a list of tags or keywords as part of the metadata. 
For waveform-to-label models, the model contributor may include an optional confidence threshold, where predictions with a probability lower than the confidence threshold are labeled as ``uncertain''. 

#### Metadata Spec


required fields:

- `sample_rate` (`int`)
    - range `(0, 396000)`
    - Model sample rate. Input tracks will be resampled to this value. 
- `domains` (`List[str]`)
    - List of data domains for the model. The list should contain any of the following strings (any others will be ignored): `["music", "speech", "environmental", "other"]`
- `short_description`(`str`)
    -  max 60 chars
    -  short description of the model. should contain a brief message with the model's purpose, e.g. "Use me for separating vocals from the background!". 
-  `long_description` (`str`)
    -  max 280 chars
    -  long description of the model. Shown in the detailed view of the model UI.
-  `tags` (`List[str]`)
    -  list of tags (to be shown in the detailed view)
    -  each tag should be 15 characters max
    -  max 5 tags per model. 
-  `labels` (`List[str`)
    -  output labels for the model. Depending on the effect type, this field means different things
    -  **waveform-to-waveform**
        -  name of each output source (e.g. `drums`, `bass`, `vocal`). To create the track name for each output source, each one of the labels will be appended to the mixture track's name.
    -  **waveform-to-labels**:
        -  labeler models should output a list of class probabilities with shape `(n_timesteps, n_class)` and a list of start/stop timestamps for each label `(n_timesteps, 2)`. The labeler effect will create a add new labels by taking the argmax of each class probability and indexing into the metadata's `labels`. 
-  `effect_type` (`str`)
    -  Target effect for this model. Must be one of `["waveform-to-waveform", "waveform-to-labels"]`. 
-  `multichannel` (`bool`)
    -  If `multichannel` is set to `true`, stereo tracks are passed to the model as multichannel audio tensors, with shape `(2, n)`. Note that this means that the input could either be a mono track with shape `(1, n)` or stereo track with shape `(2, n)`.
    -  If `multichannel` is set to `false`, stereo tracks are downmixed, meaning that the input audio tensor will always be shape `(1, n)`.

### Example - Exporting a Pretrained [Asteroid](https://github.com/asteroid-team/asteroid) model

See this [example notebook](/example.ipynb), where we serialize a pretrained ConvTasNet model for speech separation using the [Asteroid](https://github.com/asteroid-team/asteroid) source separation library.

### Example - Waveform-to-Waveform model

Here's a minimal example for a model that simply boosts volume by multiplying the incoming audio by a factor of 2. 

#### Define a model class

First, we create our model. There are no internal constraints on what the internal model architecture should be, as long as you can use `torch.jit.script` or `torch.jit.trace` to serialize it, and it is able to meet the input-output constraints specified in waveform-to-waveform and waveform-to-labels models. 

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