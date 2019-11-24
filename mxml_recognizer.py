import logging
import json
import os
from collections import Counter

import fractions
import music21
import numpy as np
import pandas as pd

from utils.song import song
from create_data_from_mxml import DataConverter


# formatter = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
#                               datefmt="%Y-%m-%d %H:%M:%S")
# dict_formatter = logging.Formatter(fmt="\t\t\t\t%(message)s")
# logger = logging.getLogger("mxml_recognizer")
# dict_logger = logging.getLogger("dict")
# dict_logger.setLevel(logging.INFO)
# logger.setLevel(logging.INFO)
# dict_handler = logging.StreamHandler()
# dict_handler.setFormatter(dict_formatter)
# handler = logging.StreamHandler()
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# dict_logger.addHandler(dict_handler)
logging.basicConfig(format="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


    
    
if __name__ == "__main__":
    csv_path = r"D:\Studies\Sem7\bachelor\mxml_recognizer\database.csv"
    df = pd.DataFrame()
    input_path = r"D:\Studies\Sem7\bachelor\mxml_recognizer\mxml_files"
    input_files = os.listdir(input_path)
    for filename in input_files:
        if os.path.isdir(input_path):
            path = os.path.join(input_path, filename)
            streams = DataConverter.extract_data_from_mxml(input_path=path)
            data_converter = DataConverter()
            data_converter.create_dictionary(streams=streams)
            flat_dict = data_converter.flatten_dictionaries()
            df = df.append(
                data_converter.create_data_frame(flat_dict=flat_dict),
                ignore_index=True
            )
    df.to_csv(csv_path, sep=";")
    