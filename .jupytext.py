"""
Script to automatically transform notebooks into py files (and vice versa), using JupyText
Used for cleaner notebook version control.

Usage: create a file watcher (preferred) or use it as a command, add *.ipynb to .gitignore
Author: @mmariani
"""

import argparse

import jupytext

# Parse the file argument
parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()
file_path = str(args.file)

# Open the file
if file_path.endswith(".py"):
    # If python file, check if header is from jupytext
    with open(file_path, "r") as f:
        first_lines = "".join((f.readline(), f.readline()))
        if first_lines != "# ---\n# jupyter:\n":
            exit()

    # If is correct, convert it to ipynb
    src = jupytext.read(file_path)
    jupytext.write(src, file_path.replace(".py", ".ipynb"))
    print("DONE")

elif file_path.endswith(".ipynb"):
    # If notebook, simply convert it to py
    src = jupytext.read(file_path)
    jupytext.write(src, file_path.replace(".ipynb", ".py"))
    print("DONE")

exit()
