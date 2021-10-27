from audacitorch.utils import save_model
import torch

import audacitorch

def test_metadata(broken_metadata, metadata):
    success, msg = audacitorch.utils.validate_metadata(broken_metadata)
    assert not success

    success, msg = audacitorch.utils.validate_metadata(metadata)
    assert success


def test_save_model(wav2wavmodel, metadata):
    from pathlib import Path

    jit_model = torch.jit.script(wav2wavmodel)
    
    path = Path('./test_save_model/')
    path.mkdir(exist_ok=True, parents=True)
    
    save_model(jit_model, metadata, path)

    assert (path / 'model.pt').exists()
    assert (path / 'metadata.json').exists()

    loaded_model = torch.jit.load(path / 'model.pt')

    for x in   audacitorch.utils.get_example_inputs():
        assert torch.allclose(loaded_model(x), jit_model(x))