#!/usr/bin/env bash
set -eu
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project-env.sh"
# Never replay the preserved completed series implicitly.
root="${1:?Provide a new retained timing evidence directory under /workspace/exports}"
case "$root" in /workspace/exports/*) ;; *) echo 'Invalid retained evidence path' >&2; exit 1;; esac
mkdir "$root"
export GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json GO_PHERENCE_POCKETTTS_VOICE_MODEL=/workspace/checkpoints/pocket-tts/languages/english/model.safetensors GO_PHERENCE_POCKET_SHARED_CORPUS=/workspace/exports/pocket-native-qualification-20261003/cpu-command/prepared.json GO_PHERENCE_POCKET_SHARED_CPU_RUN=/workspace/exports/pocket-native-qualification-20261003/cpu-command/run/step-1-634d03efd7bb7df3.run.json GO_PHERENCE_POCKET_SHARED_FULL_CPU=/workspace/exports/pocket-native-qualification-20261003/cpu-command/run/step-1-634d03efd7bb7df3.model.bin GO_PHERENCE_POCKET_RELEASED_INTEGRATED_NATIVE=1
cd /workspace/projects/go-pherence/model/pockettts
for index in 1 2 3 4 5;do
 for version in before after;do
  dir="$root/$version-$index";mkdir "$dir"
  if [ "$version" = before ];then binary=/workspace/exports/pocket-allocation-fix-20261004/integrated-native-peaks/pocket-test;else binary=/workspace/exports/pocket-allocation-fix-20261004/integrated-finite-scan/pocket-test;fi
  export GO_PHERENCE_POCKET_RELEASED_INTEGRATED_OUTPUT="$dir"
  nvidia-smi > "$dir/idle.txt";nvidia-smi --query-gpu=uuid,driver_version,memory.used,utilization.gpu --format=csv > "$dir/idle.csv"
  grep -q 'No running processes found' "$dir/idle.txt";grep -q 'GPU-cb9db783-7465-139d-900c-bc8d6b6ced7c, 610.57.04, 33 MiB, 0 %' "$dir/idle.csv"
  set +e
  /usr/bin/time -v -o "$dir/time.txt" timeout -k 5s 180s "$binary" -test.run '^TestPocketNativeReleasedIntegrated$' -test.parallel=1 -test.count=1 -test.v -test.timeout=150s -test.cpuprofile="$dir/cpu.pprof" -test.memprofile="$dir/heap.pprof" -test.memprofilerate=524288 > "$dir/run.log" 2>&1
  status=$?;set -e;echo "$status" > "$dir/status.txt"
  for sample in alloc_space alloc_objects inuse_space inuse_objects;do go tool pprof -sample_index="$sample" -top -nodecount=12 "$binary" "$dir/heap.pprof" > "$dir/$sample.txt";done
  go tool pprof -top -nodecount=12 "$binary" "$dir/cpu.pprof" > "$dir/cpu-top.txt"
  grep -E 'actual updated|EMA values|profile exit|PASS|FAIL' "$dir/run.log"
  [ "$status" = 0 ] || exit "$status"
 done
done
