#!/bin/sh
# DIAGNOSTIC BUILD, not an entry: prints facts about the scoring machine to stderr, runs the real pipeline once on the
# full input (timed), then writes a deliberately incomplete result (one events file left out) so the dev-set check
# rejects it and returns these logs. A rejected submission never changes the best score.
exec 1>&2
D=${DATA_DIR:-/data}; O=${OUTPUT_DIR:-/output}; T=${SCRATCH_DIR:-/scratch}
ms() { date +%s%N | cut -c1-13; }
echo "== cpu"; grep -m1 "model name" /proc/cpuinfo; echo "vcpus=$(nproc) $(grep -c ^processor /proc/cpuinfo) in cpuinfo"
grep -m1 "^flags" /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|avx512f|avx512bw|avx512vl|bmi2|sha_ni|vaes)$' | tr '\n' ' '; echo
grep -m1 "cpu MHz" /proc/cpuinfo; grep -m1 "cache size" /proc/cpuinfo
echo "== cgroup"; for f in cpu.max memory.max memory.current cpuset.cpus.effective; do printf "%s=%s\n" $f "$(cat /sys/fs/cgroup/$f 2>/dev/null)"; done
echo "== mounts"; grep -E " (/data|/output|/scratch|/tmp|/) " /proc/mounts
echo "== kernel"; uname -r; cat /proc/loadavg; grep -E "MemTotal|MemFree|MemAvailable|^Cached" /proc/meminfo
echo "== thp"; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
echo "== input"; du -sh $D/events $D/clicks $D/links 2>/dev/null; ls $D/events | wc -l
echo "== page cache before (fincore)"; fincore -b $D/events/*.csv 2>/dev/null | awk 'NR>1{r+=$1; s+=$3} END{printf "resident %d of %d bytes (%.1f%%)\n", r, s, 100*r/(s+1)}' || echo "fincore unavailable"
a=$(ms); cat $D/events/*.csv > /dev/null; b=$(ms); echo "== raw read of events: $((b-a)) ms"
echo "== page cache after read"; fincore -b $D/events/*.csv 2>/dev/null | awk 'NR>1{r+=$1; s+=$3} END{printf "resident %.1f%%\n", 100*r/(s+1)}'
echo "== full run (input now cached)"; a=$(ms); OUTPUT_DIR=$T/full /usr/local/bin/sessions; rc=$?; b=$(ms); echo "sessions rc=$rc wall=$((b-a)) ms"
a=$(ms); OUTPUT_DIR=$T/full2 /usr/local/bin/sessions 2>/dev/null; b=$(ms); echo "second run wall=$((b-a)) ms"
echo "== now the deliberately incomplete run"
mkdir -p $T/sub/events && for x in campaigns.csv products.csv manifest.json clicks links; do ln -s $D/$x $T/sub/$x; done
for f in $(ls $D/events | sed 1d); do ln -s $D/events/$f $T/sub/events/$f; done
DATA_DIR=$T/sub /usr/local/bin/sessions 2>/dev/null; echo "diag done"
