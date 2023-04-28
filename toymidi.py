from typing import Tuple
import torch
import torch.nn as nn


class MyMidiModel(nn.Module):

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        (midi_bytes, time_stamps) = x

        for i in range(midi_bytes.shape[0]):
            if (midi_bytes[i, 0] >> 4) == 0b1001 or (midi_bytes[i, 0] >> 4) == 0b1000:
                midi_bytes[i, 1] += 12

        y = (midi_bytes, time_stamps)

        return y


# get our model
model = MyMidiModel()

# option 1: torch.jit.script
# using torch.jit.script is preferred for most cases,
# but may require changing a lot of source code
serialized_model = torch.jit.script(model)

# save!
torch.jit.save(serialized_model, 'model.pt')
