import math
import torch
import torchaudio
import pyworld

from typing import Optional, Union, Tuple, List

################################################################################
# Pretrained pitch/periodicity extractors
################################################################################

class Pitch(torch.nn.Module):
    """
    Extract pitch and periodicity estimates from each frame of audio.
    """

    def __init__(self,
                 sample_rate: int,
                 hop_length: int,
                 periodicity_mask_thresh: Optional[float] = 0.1,
                 interpolate_unvoiced_thresh: Optional[float] = None,
                 f_min: float = 30.,
                 f_max: float = 1000.,
                 pad: bool = True,
                 **kwargs
                 ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.periodicity_mask_thresh = periodicity_mask_thresh
        self.interpolate_unvoiced_thresh = interpolate_unvoiced_thresh

        self.encoder = FCNF0Encoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            pad=pad,
            f_min=f_min,
            f_max=f_max,
            **kwargs
        )

    @staticmethod
    def _ensure_valid_audio(x: torch.Tensor):
        """
        Convert inputs to mono (single-channel) audio if necessary

        Parameters:
        -----------
        x (Tensor):   audio signal

        Returns:
        --------
        out (Tensor): audio signal, shape (n_batch, signal_length)
        """
        assert 1 <= x.ndim <= 3

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            pass
        if x.ndim == 3:
            x = x.mean(1)

        return x.float()

    @staticmethod
    def _interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor):
        """
        1D linear interpolation for monotonically increasing sample points.
        """
        # Handle edge cases
        if xp.shape[-1] == 0:
            return x
        if xp.shape[-1] == 1:
            return torch.full(
                x.shape,
                fp.squeeze(),
                device=fp.device,
                dtype=fp.dtype)

        # Get slope and intercept using right-side first-differences
        m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
        b = fp[:, :-1] - (m.mul(xp[:, :-1]))

        # Get indices to sample slope and intercept
        indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
        indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)
        line_idx = torch.linspace(
            0,
            indicies.shape[0],
            1,
            device=indicies.device).to(torch.long).expand(indicies.shape)

        # Interpolate
        return m[line_idx, indicies].mul(x) + b[line_idx, indicies]

    def _interpolate_unvoiced(self,
                              pitch: torch.Tensor,
                              periodicity: torch.Tensor):
        """
        Fill pitches in unvoiced regions via linear interpolation
        """
        voiced = periodicity > self.interpolate_unvoiced_thresh

        # Handle no voiced frames
        if not voiced.any():
            return pitch

        # Pitch is linear in base-2 log-space
        pitch = torch.log2(pitch)
        pitch[~voiced] = self._interpolate(
            torch.where(~voiced[0])[0][None],
            torch.where(voiced[0])[0][None],
            pitch[voiced][None]
        ).squeeze(0)

        return 2 ** pitch

    def forward(self, x: torch.Tensor):
        """
        Produce pitch and periodicity estimates given waveform audio.

        Parameters
        ----------
        x (Tensor):           waveform audio, shape (n_batch, [n_channels], signal_length)

        Returns
        -------
        pitch (Tensor):       pitch estimate, shape (n_batch, n_frames)

        periodicity (Tensor): periodicity estimate, shape (n_batch, n_frames)
        """

        # Ensure valid audio shape (n_batch, signal_length)
        x = self._ensure_valid_audio(x)

        # Process instances individually
        n_batch = x.shape[0]

        pitch, periodicity = [], []
        for i in range(n_batch):

            _pitch, _periodicity = self.encoder(x[i])

            # Optionally, linearly interpolate pitch over unvoiced regions
            if self.interpolate_unvoiced_thresh is not None:
                _pitch = self._interpolate_unvoiced(
                    _pitch.unsqueeze(0),
                    _periodicity.unsqueeze(0)
                ).squeeze(0)

            pitch.append(_pitch)
            periodicity.append(_periodicity)

        pitch = torch.stack(pitch, dim=0)
        periodicity = torch.stack(periodicity, dim=0)

        # Optionally, apply periodicity mask to zero-out pitch estimates over
        # aperiodic regions
        if self.periodicity_mask_thresh is not None:
            mask = periodicity < self.periodicity_mask_thresh
            pitch = pitch.masked_fill_(mask, 0.)

        return pitch, periodicity


class FCNF0(torch.nn.Module):
    """
    Fully-convolutional network for pitch prediction.
    """

    def __init__(self):
        super().__init__()

        layers = ()
        layers += (
            Block(1, 256, 481, (2, 2)),
            Block(256, 32, 225, (2, 2)),
            Block(32, 32, 97, (2, 2)),
            Block(32, 128, 66),
            Block(128, 256, 35),
            Block(256, 512, 4),
            torch.nn.Conv1d(512, 1440, 4))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor):
        return self.layers(frames[:, :, 16:-15])


class Block(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            length: int = 1,
            pooling: Union[int, Tuple[int, int]] = None,
            kernel_size: int = 32
    ):
        super().__init__()

        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU())

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add normalization
        layers += (torch.nn.LayerNorm((out_channels, length)),)

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class FCNF0Encoder(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 16000,
                 hop_length: int = 128,
                 f_min: float = 31,
                 f_max: float = 1984,
                 pad: bool = True,
                 decoder: str = "locally_normal",
                 periodicity: str = "entropy"
                 ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad

        assert decoder in ["argmax", "locally_normal"]
        self.decoder = decoder

        assert periodicity in ["entropy", "max"]
        self.periodicity = periodicity

        # Resampling
        self.penn_sample_rate = 8000
        self.resample = torchaudio.transforms.Resample(
            self.sample_rate,
            self.penn_sample_rate
        ) if self.sample_rate != self.penn_sample_rate else torch.nn.Identity()

        self.model = FCNF0()
        self.model.eval()

    ##############################################
    # Post-processing
    ##############################################

    def _postprocess(self, logits: torch.Tensor):
        """
        Convert model output to pitch and periodicity
        """

        # Convert frequency range to pitch bin range
        minidx = self._frequency_to_bins(
            torch.as_tensor(
                [self.f_min],
                dtype=logits.dtype,
                device=logits.device)
        ).long()
        maxidx = self._frequency_to_bins(
            torch.as_tensor(
                [self.f_max],
                dtype=logits.dtype,
                device=logits.device),
            "ceil"
        ).long()

        # Remove frequencies outside of allowable range
        mask_empty = torch.arange(
            logits.shape[1]
        )[None, :, None].to(logits.device)
        mask_min = mask_empty < minidx
        mask_max = mask_empty >= maxidx

        logits = logits.masked_fill(
            mask_min, -float('inf')
        ).masked_fill(mask_max, -float('inf'))

        logits = logits.float()

        # Decode pitch from logits
        if self.decoder == 'argmax':
            bins, pitch = self._argmax(logits)
        elif self.decoder == 'locally_normal':
            bins, pitch = self._locally_normal(logits)
        else:
            raise ValueError(f'Decoder method {self.decoder} is not defined')

        # Decode periodicity from logits
        if self.periodicity == 'entropy':
            periodicity = self._entropy(logits)
        elif self.periodicity == 'max':
            periodicity = self._max(logits)
        else:
            raise ValueError(
                f'Periodicity method {self.periodicity} is not defined')

        return bins.T, pitch.T, periodicity.T

    ##############################################
    # Conversions
    ##############################################

    @staticmethod
    def _cents_to_bins(cents, quantize_fn: str = "floor"):
        """Convert cents to pitch bins"""

        assert quantize_fn in ["floor", "ceil"]
        if quantize_fn == "floor":
            bins = torch.floor(cents / 5).long()
        else:
            bins = torch.ceil(cents / 5).long()

        bins[bins < 0] = 0
        bins[bins >= 1440] = 1440 - 1
        return bins

    @staticmethod
    def _bins_to_cents(bins):
        """Convert pitch bins to cents"""
        return 5 * bins

    @staticmethod
    def _frequency_to_cents(frequency):
        """Convert frequency in Hz to cents"""
        return 1200 * torch.log2(frequency / 31)

    @staticmethod
    def _cents_to_frequency(cents):
        """Converts cents to frequency in Hz"""
        return 31 * 2 ** (cents / 1200)

    def _frequency_to_bins(self, frequency, quantize_fn: str = "floor"):
        """Convert frequency in Hz to pitch bins"""
        return self._cents_to_bins(
            self._frequency_to_cents(frequency), quantize_fn)

    def _bins_to_frequency(self, bins):
        """Converts pitch bins to frequency in Hz"""
        return self._cents_to_frequency(self._bins_to_cents(bins))

    ##############################################
    # Pitch decoding
    ##############################################

    def _argmax(self, logits: torch.Tensor):
        """Decode pitch using argmax"""
        # Get pitch bins
        bins = logits.argmax(dim=1)

        # Convert to hz
        pitch = self._bins_to_frequency(bins)
        return bins, pitch

    def _locally_normal(self, logits: torch.Tensor, window: int = 19):
        """Decode pitch using a normal assumption around the argmax"""

        # Get center bins
        bins = logits.argmax(dim=1)

        # Pad
        padded = torch.nn.functional.pad(
            logits.squeeze(2),
            (window // 2, window // 2),
            value=-float('inf')
        )

        # Get indices
        indices = bins.repeat(1, window) + torch.arange(
            window, device=bins.device
        )[None]

        # Get values in cents
        cents = self._bins_to_cents(
            torch.clip(indices - window // 2, 0)
        )
        logits = torch.gather(padded, 1, indices)

        # Get local distributions
        distributions = torch.nn.functional.softmax(logits, dim=1)

        # Pitch is expected value in cents
        pitch = distributions * cents.float()
        pitch = torch.sum(pitch, dim=1, keepdim=True)

        # Convert to hz
        pitch = self._cents_to_frequency(pitch)

        return bins, pitch

    ##############################################
    # Periodicity decoding
    ##############################################

    @staticmethod
    def _entropy(logits: torch.Tensor):
        """Entropy-based periodicity"""
        distribution = torch.nn.functional.softmax(logits, dim=1)
        return (
                1 + 1 / math.log(1440) * \
                (distribution * torch.log(distribution + 1e-7)).sum(dim=1))

    @staticmethod
    def _max(logits):
        """Periodicity as the maximum confidence"""
        return torch.nn.functional.softmax(
            logits, dim=1).max(dim=1).values

    def forward(self, x: torch.Tensor, batch_size: int = 100):
        """
        Extract pitch and periodicity estimates from a single audio signal
        """
        x = x.flatten().unsqueeze(0)

        pitch: List[torch.Tensor] = [] 
        periodicity: List[torch.Tensor] = []

        # Apply extra padding to match framing of DIO and spectrogram implementations
        pad_amt = (self.hop_length - x.shape[-1] % self.hop_length)
        x = torch.nn.functional.pad(x, (0, pad_amt))

        # Use target frame count to divide resampled audio into frames, allowing for
        # fractional / variable hop lengths
        total_frames = x.shape[-1] // self.hop_length

        # Resample audio to FCNF0-compatible sample rate
        x = self.resample(x)

        # Constants to ensure compatibility with PENN algorithms
        penn_window_size = 1024
        penn_hop_size = 80

        # Using ideal (possibly fractional) resampled hop length, determine start
        # index of each frame
        start_idxs = torch.linspace(0, x.shape[-1], total_frames + 1).long()[..., :-1]

        # Optionally center-pad audio using approximate resampled hop length
        resampled_hop_length = int(self.hop_length * self.penn_sample_rate / self.sample_rate)
        padding = int((penn_window_size - resampled_hop_length) / 2)
        if self.pad:
            x = torch.nn.functional.pad(x, (padding, padding))

        # Default to running all frames in a single batch
        batch_size = total_frames if batch_size is None else batch_size

        # Generate batches of frames
        for i in range(0, total_frames, batch_size):

            # Size of this batch
            batch = min(total_frames - i, batch_size)

            # Slice and chunk audio
            frames = torch.zeros(batch, x.shape[0], penn_window_size).to(x.device)

            for j in range(batch):
                start = start_idxs[i + j]
                end = min(start + penn_window_size, x.shape[-1])
                frames[j, :, :end - start] = x[:, start:end]

            logits = self.model(frames)

            result = self._postprocess(
                logits
            )  # bins, pitch, periodicity

            pitch.append(result[1])
            periodicity.append(result[2])

        # Concatenate results
        pitch = torch.cat(pitch, 1).squeeze(0)
        periodicity = torch.cat(periodicity, 1).squeeze(0)

        return pitch, periodicity