import logging
from collections import Counter

import music21
import numpy as np

from mxml_recognizer.utils.song import song


formatter = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
dict_formatter = logging.Formatter(fmt="\t\t\t\t%(message)s")
logger = logging.getLogger("mxml_recognizer")
dict_logger = logging.getLogger("dict")
dict_logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
dict_handler = logging.StreamHandler()
dict_handler.setFormatter(dict_formatter)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
dict_logger.addHandler(dict_handler)


class StreamAnalyzer:
    """
    Class that analyses music21.stream objects and extracts information from them
    """
    def __init__(self, stream):
        self.stream = stream
        
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
        pitch_frequencies = [note.pitch.frequency
                             for note in self.notes_in_stream]
        weighted_pitch_frequencies = [note.pitch.frequency * note.duration.quarterLength
                                      for note in self.notes_in_stream]
        notes_durations = [note.duration.quarterLength 
                           for note in self.notes_in_stream]
        self.avg_pitch_freq = np.average(pitch_frequencies)
        self.weighted_avg_pitch_freq = np.average(weighted_pitch_frequencies)
        self.pitch_std = np.std(pitch_frequencies)
        self.avg_note_duration = np.average(notes_durations)
        self.note_duration_std = np.std(notes_durations)
        self.notes_by_pitch = Counter(pitch_frequencies)
        self.notes_by_duration = Counter(notes_durations)
        
    
if __name__ == "__main__":
    stream = song()
    analyzer = StreamAnalyzer(stream=stream)
    analyzer.extract_parameters()
    logger.info(f"Average pitch frequency: {analyzer.avg_pitch_freq:.4f} Hz")
    logger.info("Weighted average pitch frequency (by note duration): "
                f"{analyzer.weighted_avg_pitch_freq:.4f} Hz")
    logger.info(f"Pitch standard deviation: {analyzer.pitch_std:.4f} Hz")
    logger.info(f"Average note duration: {analyzer.avg_note_duration:.4f} quarter length")
    logger.info(f"Note duration standard deviation: {analyzer.note_duration_std:.4f} quarter length")
    logger.info("Number of notes by pitch:")
    for pitch, amount in analyzer.notes_by_pitch.items():
        dict_logger.info(f"Pitch: {pitch:.4f} Hz, amount: {amount}", )
    logger.info(f"Number of notes by duration:")
    for duration, amount in analyzer.notes_by_duration.items():
        dict_logger.info(f"Duration: {duration} quarter length, amount: {amount}")
    stream.show(app="C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe")
    stream.show("midi")