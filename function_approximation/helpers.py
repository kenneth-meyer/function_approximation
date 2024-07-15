"""

    Helpers for function approximation repo
    
    Things include: parsing input files, amongst other things
"""

import yaml

def parse_nn_yaml(input_file):
    """ Parses YAML file for NN training

    parses a YAML file knowing that it's supposed to be for a NN training thing,
    so throw errors accorindly/look for certain things in the input file

    Parameters
    ----------
    input_file : str
        filepath to the yaml file
    """

    with open(input_file) as stream:
        try:
            input_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return input_dict
