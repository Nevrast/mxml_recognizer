import music21
import fractions

from collections import Counter

import numpy as np


class StreamAnalyzer:
    """
    Class that analyses music21.stream objects and extracts information from them
    """
    def __init__(self, stream, era: str):
        self.stream = stream
        self.era = era
        
        self.notes_in_stream = []
        self._flatten_stream()
        
        self.avg_pitch_freq = None
        self.weighted_avg_pitch_freq = None
        self.pitch_std = None
        self.avg_note_duration = None
        self.note_duration_std = None
        self.notes_by_pitch = None
        self.notes_by_duration = None
        
    def _flatten_stream(self) -> None:
        """
        Makes stream "flat" which gives access to all embedded streams
        """
        flat_stream = self.stream.flat
        for note in flat_stream.notes:
            if isinstance(note, music21.chord.Chord):
                for note_in_chord in note.notes:
                    self.notes_in_stream.append(note_in_chord)
            else:
                self.notes_in_stream.append(note)

    def extract_parameters(self) -> None:
        """
        Method that calculates parameters from list of notes
        """
        pitch_frequencies = [note.pitch.frequency
                             for note in self.notes_in_stream]
        weighted_pitch_frequencies = [note.pitch.frequency * note.duration.quarterLength
                                      for note in self.notes_in_stream]
        notes_durations = [note.duration.quarterLength
                           if not isinstance(note.duration.quarterLength, fractions.Fraction)
                           else note.duration.quarterLength.numerator / note.duration.quarterLength.denominator
                           for note in self.notes_in_stream]
        self.avg_pitch_freq = np.average(pitch_frequencies)
        self.weighted_avg_pitch_freq = np.average(weighted_pitch_frequencies)
        self.pitch_std = np.std(pitch_frequencies)
        self.avg_note_duration = np.average(notes_durations)
        self.note_duration_std = np.std(notes_durations)
        self.notes_by_pitch = Counter(pitch_frequencies)
        self.notes_by_duration = Counter(notes_durations)
        self.name = self.stream.filePath.stem