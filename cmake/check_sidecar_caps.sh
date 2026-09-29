#!/bin/sh
# POST_BUILD hook for the cupti-profiler-sidecar target.
#
# Runs after every re-link. Detects whether the sidecar binary carries
# CAP_NET_ADMIN and, if not, prints a persistent multi-line reminder
# to stderr with the exact `setcap` command needed to enable SIDECAR
# mode with the taskstats backend.
#
# Non-fatal: exits 0 whether the cap is present or missing. Missing
# CAP_NET_ADMIN only blocks SIDECAR-mode runtime configuration; LEGACY
# (in-process /proc) mode is unaffected.
#
# $SIDECAR is the absolute path to the freshly-linked binary,
# injected via CMake's -E env wrapper in the caller.

set -u

if [ -z "${SIDECAR:-}" ] || [ ! -x "$SIDECAR" ]; then
    echo "[cupti-profiler] check_sidecar_caps: SIDECAR not set or not executable" >&2
    exit 0
fi

# `getcap` is in libcap2-bin (Debian/Ubuntu) / libcap (RHEL). Absence
# is not fatal — we fall through to a "please install libcap" hint.
if ! command -v getcap >/dev/null 2>&1; then
    cat >&2 <<EOF

===============================================================================
[cupti-profiler] getcap not found — cannot verify sidecar capabilities.
  Install libcap2-bin (Debian/Ubuntu) or libcap (RHEL) to enable
  automatic capability checking. Or grant the cap manually:

    sudo setcap cap_net_admin=ep $SIDECAR

  Then verify:
    getcap $SIDECAR

  Required only for SIDECAR mode + taskstats backend; LEGACY (/proc,
  in-process) mode needs no capability.
===============================================================================
EOF
    exit 0
fi

caps=$(getcap "$SIDECAR" 2>/dev/null || true)
if echo "$caps" | grep -q 'cap_net_admin'; then
    echo "[cupti-profiler] sidecar has cap_net_admin: $SIDECAR"
    exit 0
fi

cat >&2 <<EOF

===============================================================================
[cupti-profiler] Sidecar binary lacks CAP_NET_ADMIN.

  Required for SIDECAR mode with the taskstats backend. To grant:

    sudo setcap cap_net_admin=ep $SIDECAR

  Verify with:
    getcap $SIDECAR

  Must be re-run after every rebuild (linking strips file capabilities).
  Alternative: run the workload as root — implicit CAP_NET_ADMIN.
  Without the cap, SIDECAR mode Configure() will fail at startup;
  LEGACY mode (in-process /proc reads) continues to work unchanged.
===============================================================================
EOF
exit 0
