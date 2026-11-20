#!/bin/sh
# Prints wall seconds of the release binary on the 20M-event practice-size set (last line = score).
set -eu
G=/private/tmp/claude-501/-Users-cheryl--paradev-worktrees-builders-hackathon-organic-recess-5957a029/a1163b9f-fd93-434f-97fd-980b9bf5b471/scratchpad
cargo build --release -q
O=$(mktemp -d "$G/bench.XXXX")
s=$(python3 -c 'import time;print(time.perf_counter())')
DATA_DIR=$G/data/p20 OUTPUT_DIR=$O NCPU=8 target/release/sessions 2>/dev/null
e=$(python3 -c 'import time;print(time.perf_counter())')
rm -rf "$O"
python3 -c "print(round($e-$s,3))"
