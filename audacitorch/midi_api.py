from enum import Enum
import torch


# See https://docs.juce.com/master/tutorial_midi_message.html


class MidiMessage(Enum):
    NoteOn = 0
    NoteOff = 1
    ProgramChange = 2
    ControllerEvent = 3


"""
NOTE-ON
"""
def encodeNoteOnToken(channel : int, noteNumber : int, velocity : int):
    num_channels = 16
    num_notes = 128
    num_velocities = 128

    assert 1 <= channel <= num_channels
    assert 0 <= noteNumber <= (num_notes - 1)
    assert 0 <= velocity <= (num_velocities - 1)

    token = velocity + noteNumber * num_velocities + (channel - 1) * num_notes * num_velocities

    return token

def getMinNoteOnToken():

    minToken = 0

    return minToken

def getMaxNoteOnToken():
    num_channels = 16
    num_notes = 128
    num_velocities = 128

    maxToken = num_channels * num_notes * num_velocities - 1

    return maxToken

def isNoteOnToken(token):

    isToken = getMinNoteOnToken() <= token <= getMaxNoteOnToken()

    return isToken

def decodeNoteOnToken(token):
    num_channels = 16
    num_notes = 128
    num_velocities = 128

    assert isNoteOnToken(token)

    channel = token // (num_notes * num_velocities)
    noteNumber = (token - channel * num_notes * num_velocities) // num_velocities
    velocity = token - channel * num_notes * num_velocities - noteNumber * num_velocities

    channel += 1

    return channel, noteNumber, velocity

"""
NOTE-OFF
"""
def encodeNoteOffToken(channel : int, noteNumber : int, velocity : int):
    num_channels = 16
    num_notes = 128
    num_velocities = 128

    assert 1 <= channel <= num_channels
    assert 0 <= noteNumber <= (num_notes - 1)
    assert 0 <= velocity <= (num_velocities - 1)

    token = getMaxNoteOnToken() + 1
    token += velocity + noteNumber * num_velocities + (channel - 1) * num_notes * num_velocities

    return token

def getMinNoteOffToken():

    minToken = getMaxNoteOnToken() + 1

    return minToken

def getMaxNoteOffToken():
    num_channels = 16
    num_notes = 128
    num_velocities = 128

    maxToken = getMinNoteOffToken() + num_channels * num_notes * num_velocities - 1

    return maxToken

def isNoteOffToken(token):

    isToken = getMinNoteOffToken() <= token <= getMaxNoteOffToken()

    return isToken

def decodeNoteOffToken(token):

    assert isNoteOffToken(token)

    channel, noteNumber, velocity = decodeNoteOnToken(token - getMinNoteOffToken())

    return channel, noteNumber, velocity

"""
PROGRAM-CHANGE
"""
def encodeProgramChangeToken(channel : int, programNumber : int):
    num_channels = 16
    num_programs = 128

    assert 1 <= channel <= num_channels
    assert 0 <= programNumber <= (num_programs - 1)

    token = getMaxNoteOffToken() + 1
    token += programNumber + (channel - 1) * num_programs

    return token

def getMinProgramChangeToken():

    minToken = getMaxNoteOffToken() + 1

    return minToken

def getMaxProgramChangeToken():
    num_channels = 16
    num_programs = 128

    maxToken = getMinProgramChangeToken() + num_channels * num_programs - 1

    return maxToken

def isProgramChangeToken(token):

    isToken = getMinProgramChangeToken() <= token <= getMaxProgramChangeToken()

    return isToken

def decodeProgramChangeToken(token):
    num_channels = 16
    num_programs = 128

    assert isProgramChangeToken(token)

    relative_token = token - getMinProgramChangeToken()

    channel = relative_token // num_programs
    programNumber = relative_token - channel * num_programs

    channel += 1

    return channel, programNumber

"""
CONTROLLER-EVENT
"""
def encodeControllerEventToken(channel : int, controllerType : int, value : int):
    num_channels = 16
    num_controllers = 128
    num_values = 128

    assert 1 <= channel <= num_channels
    assert 0 <= controllerType <= (num_controllers - 1)
    assert 0 <= value <= (num_values - 1)

    token = getMaxProgramChangeToken() + 1
    token += value + controllerType * num_values + (channel - 1) * num_controllers * num_values

    return token

def getMinControllerEventToken():

    minToken = getMaxProgramChangeToken() + 1

    return minToken

def getMaxControllerEventToken():
    num_channels = 16
    num_controllers = 128
    num_values = 128

    maxToken = getMinControllerEventToken() + num_channels * num_controllers * num_values - 1

    return maxToken

def isControllerEventToken(token):

    isToken = getMinControllerEventToken() <= token <= getMaxControllerEventToken()

    return isToken

def decodeControllerEventToken(token):

    assert isControllerEventToken(token)

    channel, controllerType, value = decodeNoteOnToken(token - getMinControllerEventToken())

    return channel, controllerType, value

"""
DE-TOKENIZE
"""
@staticmethod
@torch.jit.script
def decodeToken(token):
    message_type, channel, number, value = -1, -1, -1, -1

    if isNoteOnToken(token):
        message_type = MidiMessage.NoteOn.value

        channel, number, value = decodeNoteOnToken(token)
    elif isNoteOffToken(token):
        message_type = MidiMessage.NoteOff.value

        channel, number, value = decodeNoteOffToken(token)
    elif isProgramChangeToken(token):
        message_type = MidiMessage.ProgramChange.value

        channel, number = decodeProgramChangeToken(token)
    elif isControllerEventToken(token):
        message_type = MidiMessage.ControllerEvent.value

        channel, number, value = decodeControllerEventToken(token)
    else:
        # TODO - token not supported, throw error
        pass

    return message_type, channel, number, value
