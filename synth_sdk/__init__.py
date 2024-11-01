"""Synth SDK initialization file"""

# Import version from package metadata
from importlib import metadata

try:
    __version__ = metadata.version("synth-sdk")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

# You can also add other package-level imports and initialization here
