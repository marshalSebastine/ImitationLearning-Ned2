"""
Print the structure of one frozen-feature probe result file (read only).

Usage: python inspect_probe_json.py <path to json>
"""

import json
import sys


def describe(value, depth = 0, maximum_depth = 3):
    """
    Print the keys of nested dictionaries and the length of lists.
    """
    indent = "  " * depth
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and depth < maximum_depth:
                print(f"{indent}{key}: {type(item).__name__}")
                describe(item, depth = depth + 1, maximum_depth = maximum_depth)
            else:
                text = repr(item)
                if len(text) > 80:
                    text = text[:80] + "..."
                print(f"{indent}{key} = {text}")
    elif isinstance(value, list):
        print(f"{indent}[list of {len(value)}]")
        if value and depth < maximum_depth:
            describe(value[0], depth = depth + 1, maximum_depth = maximum_depth)


if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding = "utf-8") as handle:
        content = json.load(handle)
    describe(content)
