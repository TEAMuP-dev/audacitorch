import audacitorch
from audacitorch.utils import get_example_inputs

def test_wav2wav(wav2wavmodel):
    outputs = [wav2wavmodel(x) for x in get_example_inputs()]