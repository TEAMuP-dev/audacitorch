import torch


class Resampler(torch.jit.ScriptModule):

    def forward(self, audio: torch.Tensor, sample_rate_in: int, sample_rate_out: int):
        return resample(audio, sample_rate_in, sample_rate_out)


if __name__ == "__main__":
    # create a model
    model = Resampler()

    # create a dummy input
    audio = torch.rand(1, 1, 1000)
    resampled = model(audio, 44100, 48000)
    print(audio.shape)
    print(resampled.shape)

    # save the model
    torch.jit.save(model, "resampler.pt")
