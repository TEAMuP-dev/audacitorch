import torch
import torch.nn as nn
from torchaudio.functional import detect_pitch_frequency


# TODO - see https://pytorch.org/docs/stable/jit.html
#        ... lots of good information here
#        ... lots of ways to improve this (e.g. add better docs, hints, etc.)

class MyPitchModel(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do the neural net magic!
        x = detect_pitch_frequency()

        # TODO - compute spectrogram
        # TODO - perform peak-picking and thresholding
        # TODO - end up with frame-level pitch activations

        return x


from audacitorch import WaveformToMidiBase


class MyVolumeModelWrapper(WaveformToMidiBase):

    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        # do any preprocessing here!
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        x = torch.mean(x, dim=0)

        output = self.model(x)

        # TODO - convert to list of notes (pitch, onset, offset, velocity)
        # TODO - convert to list of MIDI messages (https://mido.readthedocs.io/en/latest/messages.html#converting-to-bytes)

        # do any postprocessing here!
        # the return value should be TODO

        return output


metadata = {
    'sample_rate': 48000,
    'domain_tags': ['music', 'speech', 'environmental'],
    'short_description': 'Use me to boost volume by 3dB :).',
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
model = MyPitchModel()

# wrap it
wrapper = MyVolumeModelWrapper(model)

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
# TODO - see https://github.com/cwitkowitz/piano-transcription/blob/main/scripts/export.py
#        for ideas about a better verification methodology
test_run(wrapper)
test_run(serialized_model)

# check that we created our metadata correctly
success, msg = validate_metadata(metadata)
assert success

# save!
save_model(serialized_model, metadata, root)