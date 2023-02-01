import torch
import torch.nn as nn


class MyMidiModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv1d(1, 256, 1024, stride=512)
        self.bn0 = nn.BatchNorm1d(256)

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, (3, 3), padding="same")
        self.relu = nn.ReLU()

        self.linear = nn.Linear(256, 88)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        x = torch.transpose(x, 2, 3)

        # input is (batch, channels, time, freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # (batch, 1, time, freq)

        x = x.squeeze(1)

        x = self.linear(x)

        x = torch.transpose(x, 1, 2)

        return x


from audacitorch import WaveformToMidiBase


class MyMidiModelWrapper(WaveformToMidiBase):

    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        # do any preprocessing here!
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        output = self.model(x.unsqueeze(0)).squeeze(0)

        output = torch.sigmoid(output)

        threshold = torch.quantile(output, 0.95)

        output[output < threshold] = 0
        output[output >= threshold] = 1

        #num_frames = 1 + len(x) // self.hop_length
        #times = torch.arange(num_frames) * self.hop_length / self.sample_rate

        notes = self.salience_to_notes(output, 0.)

        # TODO - convert to list of MIDI messages (https://mido.readthedocs.io/en/latest/messages.html#converting-to-bytes)

        # do any postprocessing here!
        # the return value should be TODO

        return output


metadata = {
    'sample_rate': 48000,
    'domain_tags': ['music'],
    'short_description': 'Transcribe MIDI notes :).',
    'long_description': 'This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.',
    'tags': ['volume boost'],
    'labels': ['boosted'],
    'effect_type': 'waveform-to-midi',
    'multichannel': False,
}

from pathlib import Path
from audacitorch.utils import save_model, validate_metadata, \
    get_example_inputs, test_run

# create a root dir for our model
root = Path('booster-net')
root.mkdir(exist_ok=True, parents=True)

# get our model
model = MyMidiModel()

# wrap it
wrapper = MyMidiModelWrapper(model)

# serialize it using torch.jit.script, torch.jit.trace,
# or a combination of both.

# option 1: torch.jit.script
# using torch.jit.script is preferred for most cases,
# but may require changing a lot of source code
serialized_model = torch.jit.script(wrapper)

# option 2: torch.jit.trace
# using torch.jit.trace is typically easier, but you
# need to be extra careful that your serialized model behaves
# properly after tracing
#example_inputs = get_example_inputs()
#serialized_model = torch.jit.trace(wrapper, example_inputs[0],
#                                   check_inputs=example_inputs)

# take your model for a test run!
test_run(wrapper)
test_run(serialized_model)

# check that we created our metadata correctly
success, msg = validate_metadata(metadata)
assert success

# save!
save_model(serialized_model, metadata, root)