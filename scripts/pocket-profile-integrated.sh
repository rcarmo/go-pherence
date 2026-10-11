#!/usr/bin/env bash
set -eu
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project-env.sh"
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:-$PROJECT_RUN_DIR/integrated}"
case "$root" in "$PROJECT_RUN_DIR"/*) ;; *) echo 'Qualification scratch must be under PROJECT_RUN_DIR' >&2; exit 1;; esac
mkdir "$root"
root=$(cd -- "$root" && pwd -P)
run_root=$(cd -- "$PROJECT_RUN_DIR" && pwd -P)
case "$root" in "$run_root"/*) ;; *) echo 'Qualification scratch escapes run directory' >&2; exit 1;; esac
export GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1
timeout -k 5s 90s go vet -p=1 ./backends/training ./model/pockettts > "$root/vet.log" 2>&1
timeout -k 5s 90s go test -c -p=1 ./model/pockettts -o "$PROJECT_RUN_DIR/pocket-test" > "$root/compile.log" 2>&1
mv "$PROJECT_RUN_DIR/pocket-test" "$root/pocket-test" # keep one binary until analysis
unset GO_PHERENCE_DISABLE_VULKAN
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json GO_PHERENCE_POCKETTTS_VOICE_MODEL=/workspace/checkpoints/pocket-tts/languages/english/model.safetensors GO_PHERENCE_POCKET_SHARED_CORPUS=/workspace/exports/pocket-native-qualification-20261003/cpu-command/prepared.json GO_PHERENCE_POCKET_SHARED_CPU_RUN=/workspace/exports/pocket-native-qualification-20261003/cpu-command/run/step-1-634d03efd7bb7df3.run.json GO_PHERENCE_POCKET_SHARED_FULL_CPU=${GO_PHERENCE_POCKET_SHARED_FULL_CPU:-/workspace/exports/pocket-native-qualification-20261003/cpu-command/run/step-1-634d03efd7bb7df3.model.bin} GO_PHERENCE_POCKET_RELEASED_INTEGRATED_NATIVE=1 GO_PHERENCE_POCKET_RELEASED_INTEGRATED_OUTPUT="$root"
cd model/pockettts
{ printf 'run=%s\nTMPDIR=%s\nGOCACHE=%s\nGOMODCACHE=%s\nmemprofilerate=%s\n' "$PROJECT_RUN_DIR" "$TMPDIR" "$GOCACHE" "$GOMODCACHE" "${POCKET_PROFILE_MEM_RATE:-1}"; go version;git rev-parse HEAD; } > "$root/environment.txt"
timeout -k 2s 15s nvidia-smi > "$root/idle.txt"
timeout -k 2s 15s nvidia-smi --query-gpu=uuid,driver_version,memory.used,utilization.gpu --format=csv > "$root/idle.csv"
# Hardware/runtime provenance remains mandatory. Sharing is an explicit
# operator-approved mode, not a fallback after a failed idle gate.
grep -q '^GPU-cb9db783-7465-139d-900c-bc8d6b6ced7c, 610.57.04,' "$root/idle.csv"
if [ "${POCKET_PROFILE_SHARED_GPU:-0}" = 1 ]; then
 printf '%s\n' 'Shared GPU authorised by Rui on 2026-10-05; correctness/resource evidence only, not controlled timing.' > "$root/gpu-admission.txt"
else
 grep -q 'No running processes found' "$root/idle.txt"
 grep -q 'GPU-cb9db783-7465-139d-900c-bc8d6b6ced7c, 610.57.04, 33 MiB, 0 %' "$root/idle.csv"
 printf '%s\n' 'Exclusive idle baseline; timing still subject to workload/profiling restrictions.' > "$root/gpu-admission.txt"
fi
set +e
/usr/bin/time -v -o "$root/time.txt" timeout -k 5s "${POCKET_PROFILE_OUTER_TIMEOUT:-240}s" "$root/pocket-test" -test.run '^TestPocketNativeReleasedIntegrated$' -test.parallel=1 -test.count=1 -test.v -test.timeout="${POCKET_PROFILE_TEST_TIMEOUT:-210}s" -test.cpuprofile="$root/cpu.pprof" -test.memprofile="$root/heap.pprof" -test.memprofilerate="${POCKET_PROFILE_MEM_RATE:-1}" > "$root/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$root/status.txt"
timeout -k 2s 15s nvidia-smi > "$root/gpu-after.txt"
for sample in alloc_space alloc_objects inuse_space inuse_objects;do
 timeout -k 2s 30s go tool pprof -sample_index="$sample" -top -nodecount=18 "$root/pocket-test" "$root/heap.pprof" > "$root/$sample.txt"
done
timeout -k 2s 30s go tool pprof -top -nodecount=18 "$root/pocket-test" "$root/cpu.pprof" > "$root/cpu-top.txt"
cat "$root/run.log"
head -27 "$root/alloc_space.txt"
printf 'Dispose of profile/checkpoint scratch after analysis: %s\n' "$root"
exit "$status"
