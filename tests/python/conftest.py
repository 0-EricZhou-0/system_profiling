"""Pytest setup: put the staged build dir + generated proto dir on sys.path
so `import cupti_profiler` and `import events_pb2` resolve without an install.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Staged Python package: build/python/cupti_profiler/
sys.path.insert(0, os.path.join(_REPO, "build", "python"))

# Generated protobuf modules: generated/proto/{events,session_metadata,profiler_config}_pb2.py
sys.path.insert(0, os.path.join(_REPO, "generated", "proto"))
