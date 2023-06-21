import math
import torch
import torch.fft as fft
import torchaudio

import numpy as np
import librosa as li

import os
import json
import math
from pathlib import Path
from typing import Union, Dict, List
from IPython.display import Audio, display

################################################################################
# Utilities for FCNF0++ pitch estimation model
################################################################################

# Maximum & minimum frequencies to predict
FMIN = 31
FMAX = 1984

# Tuning
A4 = 440

LOW = "LOW"  # For pitches below FMIN
HIGH = "HIGH"  # For pitches above FMAX
EPS_CENTS = 50  # Tolerance for cutoffs, in cents (hundredths of a semitone)

@torch.jit.script
def f0_to_pitch(freq: Union[torch.Tensor, float], 
                f_min: float = FMIN, 
                f_max: float = FMAX,
                eps_cents: float = EPS_CENTS,
                low_label: str = LOW,
                high_label: str = HIGH,
                a4: float = A4):
    
    # Tensor conversions
    if not isinstance(freq, torch.Tensor):
        freq = torch.as_tensor([freq]).float()
    
    if not isinstance(a4, torch.Tensor):
        a4 = torch.as_tensor([a4]).float()
    c0 = a4 * math.pow(2, -4.75)
        
    f_min = torch.as_tensor([f_min]).float()
    f_max = torch.as_tensor([f_max]).float()
    
    # Use standard pitch classes
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Determine if frequency falls outside valid range
    raw_min = 12*torch.log2(freq/f_min)
    raw_max = 12*torch.log2(freq/f_max)
    
    if raw_min < -eps_cents/100:
        return low_label
    if raw_max > eps_cents/100:
        return high_label
    
    # Bound to valid range
    freq = torch.maximum(f_min, freq)
    freq = torch.minimum(f_max, freq)
    
    h = round((12*torch.log2(freq/c0)))
    octave = h // 12
    octave = octave.long().item()
    
    n = h % 12
    n = n.long().item()
    
    return pitch_classes[n] + str(octave)


# Map any floating-point pitch prediction to a pitch-class label
@torch.jit.script
def encode_output(f0: torch.Tensor):
    
    # Pitch-class labels by index
    idx_to_label : Dict[int, str] = {
        0: 'LOW', 1: 'HIGH', 2: 'B0', 3: 'C1', 4: 'C#1', 5: 'D1', 6: 'D#1', 7: 'E1', 
        8: 'F1', 9: 'F#1', 10: 'G1', 11: 'G#1', 12: 'A1', 13: 'A#1', 14: 'B1', 15: 'C2', 
        16: 'C#2', 17: 'D2', 18: 'D#2', 19: 'E2', 20: 'F2', 21: 'F#2', 22: 'G2', 23: 'G#2', 
        24: 'A2', 25: 'A#2', 26: 'B2', 27: 'C3', 28: 'C#3', 29: 'D3', 30: 'D#3', 31: 'E3', 
        32: 'F3', 33: 'F#3', 34: 'G3', 35: 'G#3', 36: 'A3', 37: 'A#3', 38: 'B3', 39: 'C4', 
        40: 'C#4', 41: 'D4', 42: 'D#4', 43: 'E4', 44: 'F4', 45: 'F#4', 46: 'G4', 47: 'G#4', 
        48: 'A4', 49: 'A#4', 50: 'B4', 51: 'C5', 52: 'C#5', 53: 'D5', 54: 'D#5', 55: 'E5', 
        56: 'F5', 57: 'F#5', 58: 'G5', 59: 'G#5', 60: 'A5', 61: 'A#5', 62: 'B5', 63: 'C6', 
        64: 'C#6', 65: 'D6', 66: 'D#6', 67: 'E6', 68: 'F6', 69: 'F#6', 70: 'G6', 71: 'G#6', 
        72: 'A6', 73: 'A#6', 74: 'B6'
    }
    label_to_idx: Dict[str, int] = {l: i for i, l in idx_to_label.items()}
    n_labels = len(label_to_idx)
    
    assert f0.ndim == 2  # (n_channels, n_frames)
    n_channels = f0.shape[0]
    n_frames = f0.shape[1]
    
    f0 = f0.float()
    
    # Prepare to one-hot encode outputs
    outs = torch.zeros(n_channels, n_frames, n_labels)
    
    for i in range(n_channels):
        for j in range(n_frames):
            idx = label_to_idx[f0_to_pitch(f0[i, j])]
            outs[i, j, idx] = 1.
    
    return outs