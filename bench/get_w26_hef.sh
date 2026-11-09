#!/usr/bin/env bash
# get_w26_hef.sh: restore W26's three compiled .hef files from git history.
#
# W26 deleted them in 79bbee1 ("FieldTesting", 2026-04-11), but git keeps them.
# They go to ~/ME450/Sandbox/hef/, outside the repo, so no binaries get committed
# (weights stay out of git, per the 11-06 snapshot §10.2).
#
#   cd ~/ME450/MainDesign && bash bench/get_w26_hef.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/ME450/Sandbox/hef}"
mkdir -p "$OUT"
git show 0f0e01c:Vision/weights/turkey_640.hef      > "$OUT/turkey_640.hef"       # 04-01, what config.py pointed at
git show 0f0e01c:Vision/weights/turkey_640_v4.hef   > "$OUT/turkey_640_v4.hef"    # 04-01 recompile
git show f31b8b5:Vision/weights/best640_hailo8l.hef > "$OUT/best640_hailo8l.hef"  # 04-04, single output
ls -l "$OUT"
sha256sum "$OUT"/*.hef 2>/dev/null || shasum -a 256 "$OUT"/*.hef
