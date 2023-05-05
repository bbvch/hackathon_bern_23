import numpy as np
import yaml
import json
import argparse
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from mlem.api import load

def evaluate_model(model, test_data, train_data):

    metric = {}

    y_pred_test = model.predict(test_data[['x']])
    metric['mae_test'] = mean_absolute_error(y_pred= y_pred_test, y_true=test_data['y'])
    metric['mape_test'] = mean_absolute_percentage_error(y_pred=y_pred_test, y_true=test_data['y'])

    y_pred_train = model.predict(train_data[['x']])
    metric['mae_train'] = mean_absolute_error(y_pred=y_pred_train, y_true=train_data['y'])
    metric['mape_train'] = mean_absolute_percentage_error(y_pred=y_pred_train,y_true=train_data['y'])

    return metric, y_pred_test, y_pred_train

def evaluate_model_from_config(config):

    #Load data from inputs
    test_data = pd.read_csv(config.evaluate.inputs.test_data)
    train_data = pd.read_csv(config.evaluate.inputs.train_data)

    model = load(config.evaluate.inputs.model)

    #Evaluate and save to outputs
    metric, y_pred_test, y_pred_train = evaluate_model(model, test_data, train_data)
    metrics_json = json.dumps(metric)

    with open(config.evaluate.outputs.metric, 'w') as fid:
        fid.write(metrics_json)

    df = pd.concat([test_data, train_data])
    df['y_pred'] = np.concatenate([y_pred_test, y_pred_train])
    df['train'] = np.concatenate([np.zeros(y_pred_test.size), np.ones(y_pred_train.size)], axis=0)
    df.to_csv(config.evaluate.outputs.result, index=False)


if __name__ == "__main__":
    print("Started evaluate stage")

    from helpers.load_config import default_config_parser, load_config_from_command_line
    args = default_config_parser()
    config = load_config_from_command_line(args.config_path)
    evaluate_model_from_config(config)

    print("Finished evaluate stage")
