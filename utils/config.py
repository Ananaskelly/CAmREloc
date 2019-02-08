import json


def read_config(path_to_json):

    with open(path_to_json, 'w') as config_file:
        config = json.load(config_file)
