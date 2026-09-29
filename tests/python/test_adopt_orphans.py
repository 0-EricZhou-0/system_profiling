"""adopt_orphans(): the opt-in subreaper helper must reap the orphans
discovery saw adopted, and must never take the exit status of the
launcher's own children.

PR_SET_CHILD_SUBREAPER cannot be undone for already-adopted orphans and
would change every later test in this pytest process, so the launcher is
a fresh Python process that reports what it saw as one JSON line.
"""

import json
import subprocess
import sys

import pytest

import session_metadata_pb2
from tracing_helpers import MODES as MODE_IDS
from tracing_helpers import PY, suite_config, system_frames, tracked

MODES = list(MODE_IDS)

# A: forks the orphan-to-be O, reports both PIDs, and exits while O lives.
A_CODE = """
import json, os, subprocess, sys, time
o = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.8)"])
print(json.dumps({"a": os.getpid(), "o": o.pid}), flush=True)
time.sleep(0.4)
os._exit(0)
"""
# The tracked root: runs A, waits for it, lingers.
TREE_CODE = """
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", sys.argv[1]]).wait()
time.sleep(2.0)
"""

LAUNCHER = """
import json, os, subprocess, sys, time
import cupti_profiler as cp

cfg = json.loads(sys.argv[1])
cp.adopt_orphans()
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()

# The launcher's own child: exits at once with a distinctive status and
# stays an unwaited zombie while the orphan below is adopted and reaped.
own = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])

# tree -> A -> O. A lives long enough for O to be discovered as A's
# child, then exits; O is adopted by this launcher, lives a little, exits.
tree = subprocess.Popen([sys.executable, "-c", %r, %r], stdout=subprocess.PIPE)
""" % (TREE_CODE, A_CODE) + """
suite.add_tracked_process(tree.pid, "tree", track_descendants=True)
ids = json.loads(tree.stdout.readline())
o = ids["o"]

def state(pid):
    try:
        text = open(f"/proc/{pid}/stat").read()
    except FileNotFoundError:
        return None
    rest = text.rsplit(")", 1)[1].split()
    return rest[0], int(rest[1])   # state, ppid

adopted_by = None
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    s = state(o)
    if s is None:
        break                        # exited AND reaped
    if s[0] != "Z":
        adopted_by = s[1]
    time.sleep(0.01)
reaped = state(o) is None

rc_own = own.wait(timeout=5)         # must still be ours to collect
rc_tree = tree.wait(timeout=10)
suite.stop()
print("RESULT " + json.dumps({"o": o, "a": ids["a"], "tree": tree.pid, "me": os.getpid(),
                  "adopted_by": adopted_by, "reaped": reaped,
                  "rc_own": rc_own, "rc_tree": rc_tree}), flush=True)
# (tagged: the library's buffered std::cout may land after this line)
"""


@pytest.mark.parametrize("mode", MODES)
def test_adopt_orphans_does_not_steal(tmp_path, mode):
    cfg = suite_config(tmp_path, mode,
                       discovery={"enabled": False, "scan_interval_ms": 50})
    run = subprocess.run([PY, "-c", LAUNCHER, json.dumps(cfg)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-4000:]
    res = json.loads(next(l for l in run.stdout.splitlines()
                          if l.startswith("RESULT "))[len("RESULT "):])
    print(res)
    # Premise: the orphan really was adopted by the launcher.
    assert res["adopted_by"] == res["me"], res
    # The helper reaped the adopted orphan (otherwise it stays a zombie
    # until the launcher exits) ...
    assert res["reaped"], f"adopted orphan {res['o']} left as a zombie"
    # ... and did not take the exit status of the launcher's own
    # children. A stolen status makes Popen.wait() see ECHILD, which
    # Python reports as 0.
    assert res["rc_own"] == 7, f"launcher's own child status stolen: rc={res['rc_own']}"
    assert res["rc_tree"] == 0

    seen = tracked(system_frames(tmp_path))
    assert seen[res["o"]].discovered and seen[res["o"]].parent_pid == res["a"]
    assert seen[res["o"]].removed

    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString((tmp_path / "session_metadata.pb").read_bytes())
    sub = [c for c in meta.situation if c.check == "child subreaper (this process)"]
    assert len(sub) == 1 and sub[0].observed == "set by adopt_orphans()", list(meta.situation)
