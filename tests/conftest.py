import pytest 
import audacitorch


@pytest.fixture
def broken_metadata():
    return {'no cool': 'information here'}

@pytest.fixture
def metadata():
    return {
    'sample_rate': 48000, 
    'domain_tags': ['music', 'speech', 'environmental'],
    'short_description': 'Use me to boost volume by 3dB :).',
    'long_description':  'This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.',
    'tags': ['volume boost'],
    'labels': ['boosted'],
    'effect_type': 'waveform-to-waveform',
    'multichannel': False,
}

@pytest.fixture
def wav2wavmodel():
    from audacitorch import WaveformToWaveformBase
    import torch
    import torch.nn as nn

    class MyVolumeModel(nn.Module):

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # do the neural net magic!
            x = x * 2

            return x

    class MyVolumeModelWrapper(WaveformToWaveformBase):
        
        def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
            
            # do any preprocessing here! 
            # expect x to be a waveform tensor with shape (n_channels, n_samples)

            output = self.model(x)

            # do any postprocessing here!
            # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
        
            return output
    
    model = MyVolumeModel()
    return MyVolumeModelWrapper(model)


