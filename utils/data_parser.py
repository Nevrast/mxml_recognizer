import logging
import json
import os

import music21
import pandas as pd

from utils.stream_analyzer import StreamAnalyzer


logger = logging.getLogger(__name__)


class DataParser:
    """
    Class that takes mxml files and creates database in form of .csv file.
    """
    def __init__(self) -> None:

        self.data_dict = {}
        
    def create_dictionary(self, streams: list) -> None:
        """
        Method that creates dictionary from data calculated by StreamAnalyzer.
        """
        
        for analyzed_stream in streams:
            self.data_dict[analyzed_stream.name] = {
                "avg_pitch": analyzed_stream.avg_pitch_freq,
                "weighted_avg_pitch_freq": analyzed_stream.weighted_avg_pitch_freq,
                "pitch_std": analyzed_stream.pitch_std,
                "avg_note_duration": analyzed_stream.avg_note_duration,
                "note_duration_std": analyzed_stream.note_duration_std,
                "notes_by_pitch": analyzed_stream.notes_by_pitch,
                "notes_by_duration": analyzed_stream.notes_by_duration,
                "class_label": analyzed_stream.era
            }
        

    def _transform_2d_data(self, key_to_transform: str) -> dict:
        """
        Method that flattens subdictionaries of data from self.data_dict
        gathering all data into lists under each key instead of many
        dictionaries with same keys.
        """
        transformed_dict = {}
        for name in self.data_dict.keys():
            for key in self.data_dict[name][key_to_transform].keys():
                transformed_dict[key] = []

        for name in self.data_dict.keys():
            for key in transformed_dict.keys():
                self._normalize_numbers_of_notes(
                    name=name, key_to_transform=key_to_transform
                )
                try:
                    transformed_dict[key].append(
                        self.data_dict[name][key_to_transform][key]
                    )
                except KeyError:
                    transformed_dict[key].append(0)
        return transformed_dict


    def _normalize_numbers_of_notes(self, name: str,
                                    key_to_transform: str) -> None:
        """
        Divides each number of notes by sum of all notes
        """
        amount_of_notes = sum(
            val for val in self.data_dict[name][key_to_transform].values()
        )
        for note, value in self.data_dict[name][key_to_transform].items():
            self.data_dict[name][key_to_transform][note] = value / amount_of_notes
            

    def flatten_dictionaries(self) -> dict:
        """
        Flattens input dictionary, operation necessary for machine learning
        algorithms.
        """
        flat_dict = {}
        subdictionaries = ["notes_by_pitch", "notes_by_duration"]
        for name in self.data_dict.keys():
            for key, val in self.data_dict[name].items():
                if key not in subdictionaries:
                    if flat_dict.get(key):
                        flat_dict[key].append(val)
                    else:
                        flat_dict[key] = [val]

        for param in subdictionaries:
            transformed_dict = self._transform_2d_data(key_to_transform=param)
            flat_dict = {**flat_dict, **transformed_dict}
            
        return flat_dict
    
    def create_data_frame(self, flat_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame(data=flat_dict)
        labels = df.pop("class_label")
        df = df.join(labels)
        return df

    @staticmethod
    def extract_data_from_mxml(input_path: str) -> list:
        analyzed_streams = []
        files = os.listdir(input_path)
        for f in files:
            filepath = os.path.join(input_path, f)
            logger.info(f"Opening and reading file: {filepath}.")
            try:
                stream = music21.converter.parse(filepath)
                logger.info(f"Successfully read file: {filepath}.")
            except music21.converter.ConverterException as e:
                logger.error(f"Failure when reading file: {filepath}. "
                             f"Error message: {e}.")
                continue
            except ZeroDivisionError as e:
                logger.error(f"Failure when reading file: {filepath}. "
                             f"Error message: {e}.")
                continue
            except Exception as e:
                logger.error(f"Failure when opening file: {filepath}. "
                             f"Error message: {e}.")
                continue
            logger.info(f"Analyzing file: {filepath}.")
            stream = StreamAnalyzer(stream=stream,
                                    era=os.path.basename(input_path))
            stream.extract_parameters()
            analyzed_streams.append(stream)
        
        return analyzed_streams
    