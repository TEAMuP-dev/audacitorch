import torch
import torch.nn as nn
from torchaudio.functional import detect_pitch_frequency, loudness
from torchaudio.transforms import Spectrogram


# TODO - see https://pytorch.org/docs/stable/jit.html
#        ... lots of good information here
#        ... lots of ways to improve this (e.g. add better docs, hints, etc.)

class MyMidiModel(nn.Module):

    def __init__(self, n_fft, sample_rate, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, normalized=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do the neural net magic!
        silence = torch.zeros(24000)
        sinusoid = torch.sin(2 * torch.pi * 440 * torch.arange(48000) / 48000)
        example_sinusoid = torch.cat((silence, sinusoid, silence))
        #x = detect_pitch_frequency(sinusoid, 48000, win_length=5, freq_low=hz_min, freq_high=hz_max)
        x = self.spec(example_sinusoid)
        x /= torch.max(x)

        x = torch.threshold(x, 0.8, 0)

        freq_bins = torch.arange(1 + self.n_fft // 2) * self.sample_rate / self.n_fft

        freq_bins = 12 * (torch.log2(freq_bins / 440.0)) + 69

        pitch_activations = torch.zeros((88, x.shape[-1]))

        # TODO - compute spectrogram
        # TODO - perform peak-picking and thresholding
        # TODO - end up with frame-level pitch activations

        return x


from audacitorch import WaveformToMidiBase


class MyMidiModelWrapper(WaveformToMidiBase):

    def __init__(self, model, sample_rate, hop_length):
        super().__init__(model)

        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        # do any preprocessing here!
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        x = torch.mean(x, dim=0)

        output = self.model(x)

        num_frames = 1 + len(x) // self.hop_length
        times = torch.arange(num_frames) * self.hop_length / self.sample_rate

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

n_fft = 2048
sample_rate = 48000
hop_length = 512

# get our model
model = MyMidiModel(n_fft, sample_rate, hop_length)

# wrap it
wrapper = MyMidiModelWrapper(model, sample_rate, hop_length)

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