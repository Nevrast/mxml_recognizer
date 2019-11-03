import pytest
import music21

from mxml_recognizer import StreamAnalyzer
from utils.song import song


@pytest.fixture
def return_analyzer():
    stream = song()
    real_stream = music21.converter.parse("mxml_files\\Virgam_virtutis_tuae_duet_Vivaldi_594.mxl")
    analyzer = StreamAnalyzer(stream=real_stream)
    analyzer.extract_parameters()
    return analyzer


def test_check_avg_pitch_freq_exist(return_analyzer):
    assert return_analyzer.avg_pitch_freq is not None, "Couldn't calculate average pitch frequency."
    
    
def test_check_weighted_avg_pitch_freq_exist(return_analyzer):
    assert return_analyzer.weighted_avg_pitch_freq is not None, "Couldn't calculate weighted average pitch frequency."


def test_check_pitch_std_exist(return_analyzer):
    assert return_analyzer.pitch_std is not None, "Couldn't calculate pitch std."


def test_check_avg_note_duration_exist(return_analyzer):
    assert return_analyzer.avg_note_duration is not None, "Couldn't calculate average note duration."


def test_check_note_duration_std_exist(return_analyzer):
    assert return_analyzer.note_duration_std is not None, "Couldn't calculate note duration std."


def test_check_notes_by_pitch_exist(return_analyzer):
    assert return_analyzer.notes_by_pitch is not None, "Couldn't sort notes by pitch."


def test_check_notes_by_duration_exist(return_analyzer):
    assert return_analyzer.notes_by_duration is not None, "Couldn't sort notes by duration."
