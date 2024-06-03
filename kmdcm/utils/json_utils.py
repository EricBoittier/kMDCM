"""
Class for common json operations
"""

from pathlib import Path
import json


def load_json(path):
    json_file = Path(path)
    with open(json_file) as f:
        data = json.load(f)
    return data


def load_experiment(path):
    data = load_json(path)
    return data
