import argparse
import yaml
import box



def default_config_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()
    return args

def load_config_from_command_line(config_path):

    #Load config
    with open(config_path, "r") as fid:
        config = yaml.safe_load(fid)

    return box.ConfigBox(config)



if __name__ == "__main__":

    args = default_config_parser()
    config = load_config_from_command_line(args.config_path)
