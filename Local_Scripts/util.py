# util.py
from pathlib import Path

# Set BASE_DIR to the directory above the current one
BASE_DIR = Path(__file__).resolve().parent.parent

# Example function that uses the base directory
def get_path(relative_path):
    return BASE_DIR / relative_path