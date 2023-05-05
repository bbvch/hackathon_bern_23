import numpy as np
import pandas as pd
import yaml
import argparse

from sklearn.model_selection import train_test_split

def split_data(data, test_fraction, **kwargs):

    train_data, test_data = train_test_split(data, test_size=test_fraction, shuffle=True)

    return train_data, test_data


def split_data_from_config(config):

    #Load data from inputs
    data = pd.read_csv(config.split.inputs.trans_data)

    #Split data
    train_data, test_data = split_data(data,
                                       **config.split.param)

    #Save the data to outputs
    test_data.to_csv(config.split.outputs.test_data, index=False)
    train_data.to_csv(config.split.outputs.train_data, index=False)


if __name__ == "__main__":
    print("Started split stage")

    from helpers.load_config import default_config_parser, load_config_from_command_line

    args = default_config_parser()
    config = load_config_from_command_line(args.config_path)
    split_data_from_config(config)

    print("Finished split stage")
