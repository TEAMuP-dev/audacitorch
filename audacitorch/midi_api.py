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

        token = MidiTokenizer.getMaxNoteOnToken() + 1
        token += velocity + noteNumber * NUM_VALUES + (channel - 1) * NUM_NUMBERS * NUM_VALUES

        return token

    @staticmethod
    def getMinNoteOffToken():

        minToken = MidiTokenizer.getMaxNoteOnToken() + 1

        return minToken

    @staticmethod
    def getMaxNoteOffToken():

        maxToken = MidiTokenizer.getMinNoteOffToken() + NUM_CHANNELS * NUM_NUMBERS * NUM_VALUES - 1

        return maxToken

    @staticmethod
    def isNoteOffToken(token):

        isToken = MidiTokenizer.getMinNoteOffToken() <= token <= MidiTokenizer.getMaxNoteOffToken()

        return isToken

    @staticmethod
    def decodeNoteOffToken(token):

        assert MidiTokenizer.isNoteOffToken(token)

        channel, noteNumber, velocity = MidiTokenizer.decodeNoteOnToken(token - MidiTokenizer.getMinNoteOffToken())

        return channel, noteNumber, velocity

    """
    PROGRAM-CHANGE
    """
    @staticmethod
    def getProgramChangeToken(channel, programNumber):

        assert 1 <= channel <= 16
        assert 0 <= programNumber <= 127

        token = MidiTokenizer.getMaxNoteOffToken() + 1
        token += programNumber + (channel - 1) * NUM_NUMBERS

        return token

    @staticmethod
    def getMinProgramChangeToken():

        minToken = MidiTokenizer.getMaxNoteOffToken() + 1

        return minToken

    @staticmethod
    def getMaxProgramChangeToken():

        maxToken = MidiTokenizer.getMinProgramChangeToken() + NUM_CHANNELS * NUM_NUMBERS - 1

        return maxToken

    @staticmethod
    def isProgramChangeToken(token):

        isToken = MidiTokenizer.getMinProgramChangeToken() <= token <= MidiTokenizer.getMaxProgramChangeToken()

        return isToken

    @staticmethod
    def decodeProgramChangeToken(token):

        assert MidiTokenizer.isProgramChangeToken(token)

        relative_token = token - MidiTokenizer.getMinProgramChangeToken()

        channel = relative_token // NUM_NUMBERS
        programNumber = relative_token - channel * NUM_NUMBERS

        channel += 1

        return channel, programNumber

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

    @staticmethod
    def getMinControllerEventToken():

        minToken = MidiTokenizer.getMaxProgramChangeToken() + 1

        return minToken

    @staticmethod
    def getMaxControllerEventToken():

        maxToken = MidiTokenizer.getMinControllerEventToken() + NUM_CHANNELS * NUM_NUMBERS * NUM_VALUES - 1

        return maxToken

    @staticmethod
    def isControllerEventToken(token):

        isToken = MidiTokenizer.getMinControllerEventToken() <= token <= MidiTokenizer.getMaxControllerEventToken()

        return isToken

    @staticmethod
    def decodeControllerEventToken(token):

        assert MidiTokenizer.isControllerEventToken(token)

        channel, controllerType, value = MidiTokenizer.decodeNoteOnToken(token - MidiTokenizer.getMinControllerEventToken())

        return channel, controllerType, value

    """
    DE-TOKENIZE
    """
    @staticmethod
    @torch.jit.script
    def decodeToken(token):
        if MidiTokenizer.isNoteOnToken(token):
            message_type = MidiMessage.NoteOn.value

            channel, number, value = MidiTokenizer.decodeNoteOnToken(token)
        elif MidiTokenizer.isNoteOffToken(token):
            message_type = MidiMessage.NoteOff.value

            channel, number, value = MidiTokenizer.decodeNoteOffToken(token)
        elif MidiTokenizer.isProgramChangeToken(token):
            message_type = MidiMessage.ProgramChange.value

            channel, number = MidiTokenizer.decodeProgramChangeToken(token)
            value = 0
        elif MidiTokenizer.isControllerEventToken(token):
            message_type = MidiMessage.ControllerEvent.value

            channel, number, value = MidiTokenizer.decodeControllerEventToken(token)
        else:
            # TODO - token not supported, throw error
            print()

        return message_type, channel, number, value
