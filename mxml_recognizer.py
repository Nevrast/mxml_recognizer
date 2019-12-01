import logging
import json
import os
from collections import Counter

import fractions

import pandas as pd

from utils.create_data_from_mxml import DataConverter
from utils.parser import cmd_parser
from utils.model import train_model


logging.basicConfig(format="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


    
if __name__ == "__main__":
    args = cmd_parser()
    if args.csv_path and args.mxml_files:
        csv_path = args.csv_path
        df = pd.DataFrame()
        input_path = args.mxml_files
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
    if args.train:
        train_model(path=args.csv_path, test_size=args.test_size, eta0=args.eta,
                    max_iter=args.max_iter, random_state=args.random_state)