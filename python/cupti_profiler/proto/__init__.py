"""Generated protobuf bindings for cupti_profiler traces and config.

protoc emits flat `import <name>_pb2` statements between sibling proto
files (e.g. system_metrics_pb2 imports tracked_process_pb2). To make
those resolve when this package is imported as `cupti_profiler.proto`,
prepend our own directory to sys.path the first time the package loads.
The cost is a single PathFinder hit at import time.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
