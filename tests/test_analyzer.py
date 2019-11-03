import pytest

from mxml_recognizer import StreamAnalyzer
from utils.song import song


def test_check_if_all_parameters_exist():
    analyzer = StreamAnalyzer(stream=song())
    analyzer.extract_parameters()
    assert analyzer.avg_pitch_freq is not None, "Couldn't calculate average pitch frequency."
    assert analyzer.weighted_avg_pitch_freq is not None, "Couldn't calculate weighted average pitch frequency."
    assert analyzer.pitch_std is not None, "Couldn't calculate pitch std."
    assert analyzer.avg_note_duration is not None, "Couldn't calculate average note duration."
    assert analyzer.note_duration_std is not None, "Couldn't calculate note duration std."
    assert analyzer.notes_by_pitch is not None, "Couldn't sort notes by pitch."
    assert analyzer.notes_by_duration is not None, "Couldn't sort notes by duration."
