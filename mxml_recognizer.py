import logging
import json
import os
from collections import Counter
from joblib import dump, load

import fractions

import pandas as pd
import numpy as np
from utils.data_parser import DataParser
from utils.parser import cmd_parser
from utils.model import train_model, classify


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
                streams = DataParser.extract_data_from_mxml(input_path=path)
                data_converter = DataParser()
                data_converter.create_dictionary(streams=streams)
                flat_dict = data_converter.flatten_dictionaries()
                df = df.append(
                    data_converter.create_data_frame(flat_dict=flat_dict),
                    ignore_index=True
                )
        df.to_csv(csv_path, sep=";")
    if args.train:
        model = train_model(path=args.csv_path, test_size=args.test_size, kernel=args.kernel,
                                gamma=args.gamma, C=args.C, random_state=args.random_state)

    if args.save_model:
        dump(model, args.save_model)
        
    if args.classify:
        if args.train:
            test_model = model
        else:
            test_model = load(args.classify)
        classify(model=test_model, data=args.csv_path)