"""
Scans through directory and prints all notebooks and the status
of each
Assumes that format from An_Introduction/Development.ipynb is followed
"""

import json
import re
import glob


def find_notebooks():
    """Finds all notebook files"""
    try:
        return glob.glob("**/*.ipynb", recursive=True)
    except TypeError:
        raise Exception("Script needs to be run with Python version >=3.5")
        

def get_status(notebook_filepath):
    """Get status of each notebook file"""
    with open(notebook_filepath, 'r') as nb_file:
        nb_json = json.load(nb_file)

    first_cell = nb_json["cells"][0]["source"][0]

    try:
        status = re.search(" \((.+)\)", first_cell).groups()[0]

    except AttributeError:
        status = "Completed"

    return status


def parse_notebooks():
    for notebook in find_notebooks():

        status = get_status(notebook)
        print("{0} [{1}]".format(notebook, status))

    return

if __name__ == "__main__":
    parse_notebooks()

