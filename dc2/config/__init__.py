"""Config files."""
from pathlib import Path

# Directory that holds the DC2-specific YAML configs shipped in this repo.
DC2_CONFIG_DIR = Path(__file__).resolve().parent

# Default config file name (without .yaml), for evaluate.py and submit.py alike.
# Switch to "dc2_edito" if running against the EDITO infrastructure.
DEFAULT_CONFIG_NAME = "dc2_wasabi"
