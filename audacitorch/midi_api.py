import torch


# See https://docs.juce.com/master/tutorial_midi_message.html
# See https://www.cs.cmu.edu/~music/cmsip/readings/MIDI%20tutorial%20for%20programmers.html


def validate_channel(channel : int):
    assert 1 <= channel <= 16


def validate_pitch(pitch : int):
    assert 0 <= pitch <= 127


def validate_velocity(velocity : int):
    assert 0 <= velocity <= 127


def encode_note_on_token(channel : int, pitch : int, velocity : int):
    validate_channel(channel)
    validate_pitch(pitch)
    validate_velocity(velocity)

    status_byte = (0b1001 << 4) + (channel - 1)

    midi_bytes = torch.ByteTensor([status_byte, pitch, velocity])

    return midi_bytes


def encode_note_off_token(channel : int, pitch : int, velocity : int):
    validate_channel(channel)
    validate_pitch(pitch)
    validate_velocity(velocity)

    status_byte = (0b1000 << 4) + (channel - 1)

    midi_bytes = torch.ByteTensor([status_byte, pitch, velocity])

    return midi_bytes


def encode_drum_note_on_token(drum_type : int, velocity : int):
    midi_bytes = encode_note_on_token(10, drum_type, velocity)

    return midi_bytes


def encode_drum_note_off_token(drum_type : int, velocity : int):
    midi_bytes = encode_note_off_token(10, drum_type, velocity)

    return midi_bytes


def validate_instrument(instrument : int):
    assert 1 <= instrument <= 128


def encode_program_change_token(channel : int, instrument : int):
    validate_channel(channel)
    validate_instrument(instrument)

    status_byte = (0b1100 << 4) + (channel - 1)

    midi_bytes = torch.ByteTensor([status_byte, instrument - 1])

    return midi_bytes


def validate_controller_type(controller_type : int):
    assert 0 <= controller_type <= 127


def validate_value(value : int):
    assert 0 <= value <= 127


def encode_controller_event_token(channel : int, controller_type : int, value : int):
    validate_channel(channel)
    validate_controller_type(controller_type)
    validate_value(value)

    status_byte = (0b1011 << 4) + (channel - 1)

    midi_bytes = torch.ByteTensor([status_byte, controller_type, value])

    return midi_bytes


def validate_bend_amt(bend_amt : int):
    assert -1. <= bend_amt <= 1.


def encode_pitch_bend_token(channel : int, bend_amt : float):
    validate_channel(channel)
    validate_bend_amt(bend_amt)

    status_byte = (0b1110 << 4) + (channel - 1)

    bend_value = round((2 ** 14) * (2 / (bend_amt + 1)))

    msb = bend_value >> 7
    lsb = bend_value - msb

    midi_bytes = torch.ByteTensor([status_byte, lsb, msb])

    return midi_bytes
