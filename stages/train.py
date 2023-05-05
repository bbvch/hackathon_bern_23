import numpy as np
import yaml
from sklearn.linear_model import Ridge
import argparse
import pandas as pd
from mlem.api import save
import pickle


def setup_train_model(alpha, data_train, **kwargs):

    model = Ridge(alpha=alpha, fit_intercept=True)

    model.fit(data_train[['x']], data_train['y'])

    return model


def setup_train_model_from_config(config):

    #1. Load data dependencies
    data = pd.read_csv(config.train.inputs.train_data)

    #2. Setup & Train Model
    model = setup_train_model(config.train.param.alpha, data)

    #3. Save model with MLEM
    save(model, config.train.outputs.model_mlem, sample_data=data[['x']])


if __name__ == "__main__":
    print("Started train stage")

    from helpers.load_config import default_config_parser, load_config_from_command_line
    args = default_config_parser()
    config = load_config_from_command_line(args.config_path)
    setup_train_model_from_config(config)

    print("Finished train stage")
