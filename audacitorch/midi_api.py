from enum import Enum
import torch


# See https://docs.juce.com/master/tutorial_midi_message.html

NUM_CHANNELS = 16
NUM_NUMBERS = 128
NUM_VALUES = 128


class MidiMessage(Enum):
    NoteOn = 0
    NoteOff = 1
    ProgramChange = 2
    ControllerEvent = 3


class MidiTokenizer:

    """
    NOTE-ON
    """
    @staticmethod
    def getNoteOnToken(channel, noteNumber, velocity):

        assert 1 <= channel <= 16
        assert 0 <= noteNumber <= 127
        assert 0 <= velocity <= 127

        token = velocity + noteNumber * NUM_VALUES + (channel - 1) * NUM_NUMBERS * NUM_VALUES

        return token

    @staticmethod
    def getMinNoteOnToken():

        minToken = 0

        return minToken

    @staticmethod
    def getMaxNoteOnToken():

        maxToken = NUM_CHANNELS * NUM_NUMBERS * NUM_VALUES - 1

        return maxToken

    @staticmethod
    def isNoteOnToken(token):

        isToken = MidiTokenizer.getMinNoteOnToken() <= token <= MidiTokenizer.getMaxNoteOnToken()

        return isToken

    @staticmethod
    def decodeNoteOnToken(token):

        assert MidiTokenizer.isNoteOnToken(token)

        channel = token // (NUM_NUMBERS * NUM_VALUES)
        noteNumber = (token - channel * NUM_NUMBERS * NUM_VALUES) // NUM_VALUES
        velocity = token - channel * NUM_NUMBERS * NUM_VALUES - noteNumber * NUM_VALUES

        channel += 1

        return channel, noteNumber, velocity

    """
    NOTE-OFF
    """
    @staticmethod
    def getNoteOffToken(channel, noteNumber, velocity):

        assert 1 <= channel <= 16
        assert 0 <= noteNumber <= 127
        assert 0 <= velocity <= 127

        token = None

        return token

    """
    PROGRAM-CHANGE
    """
    @staticmethod
    def getProgramChangeToken(channel, programNumber):

        assert 1 <= channel <= 16
        assert 0 <= programNumber <= 127

        token = None

        return token

    """
    CONTROLLER-EVENT
    """
    @staticmethod
    def getControllerEventToken(channel, controllerType, value):

        assert 1 <= channel <= 16
        assert 0 <= controllerType <= 127
        assert 0 <= value <= 127

        token = None

        return token

    """
    DE-TOKENIZE
    """
    @staticmethod
    @torch.jit.script
    def decodeToken(token):
        message_type = 0
        channel = 0
        number = 0
        value = 0

        return message_type, channel, number, value
