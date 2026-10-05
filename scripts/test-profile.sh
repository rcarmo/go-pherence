#!/usr/bin/env bash
# Per-package profiles and matching binaries; never treat tables as analysis.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project-env.sh"
args=() packages=() list_args=()
while [ "$#" -gt 0 ]; do
 case "$1" in
  -c) exec go test "${args[@]}" "${packages[@]}" "$@";;
  -o|-cpuprofile|-memprofile|-memprofilerate|-outputdir|-trace) echo 'Use PROFILE_MEM_RATE/PROFILE_EVIDENCE_ROOT; conflicting profile flags rejected' >&2; exit 2;;
  -o=*|-cpuprofile=*|-memprofile=*|-memprofilerate=*|-outputdir=*|-trace=*) echo 'Conflicting test output flags rejected' >&2; exit 2;;
  -run|-bench|-benchtime|-count|-timeout|-parallel|-p|-tags|-gcflags|-ldflags|-coverprofile|-coverpkg|-fuzz|-fuzztime|-fuzzminimizetime|-mod|-vet|-cpu)
   flag="$1";value="${2:?Missing test flag value}";args+=("$flag" "$value");if [ "$flag" = -tags ]; then list_args+=("$flag" "$value");fi;shift 2;;
  -tags=*) args+=("$1");list_args+=("$1");shift;;
  -args|--) echo 'Raw test arguments need an explicit direct profiled invocation' >&2;exit 2;;
  -*) args+=("$1");shift;;
  *) packages+=("$1");shift;;
 esac
done
[ "${#packages[@]}" -gt 0 ] || packages=(.)
# Portable retained artifacts: workspace exports, or explicit CI artifact root.
# Evidence never belongs to disposable project cache/build/runs cleanup.
if [ -n "${PROFILE_EVIDENCE_ROOT:-}" ];then
 root="$PROFILE_EVIDENCE_ROOT"
elif [ -d /workspace/exports ] && [ -w /workspace/exports ];then
 root="/workspace/exports/go-pherence-test-profiles/$GO_PHERENCE_RUN_ID"
elif [ -n "${GITHUB_WORKSPACE:-}" ];then
 root="$GITHUB_WORKSPACE/artifacts/go-pherence-test-profiles/$GO_PHERENCE_RUN_ID"
else
 root="$PWD/artifacts/go-pherence-test-profiles/$GO_PHERENCE_RUN_ID"
fi
case "$root" in /*) ;; *) echo 'Profiles require absolute retained artifact path' >&2;exit 2;; esac
case "$root" in "$PROJECT_TMP_ROOT"|"$PROJECT_TMP_ROOT"/*) echo 'Retained profiles cannot be under disposable root' >&2;exit 2;; esac
[ ! -L "$root" ] || { echo 'Symlink profile root rejected' >&2;exit 2; }
mkdir -p "$root"
# Unique invocation subdirectory permits several Make recipes/packages safely.
invocation=$(mktemp -d "$root/invocation-XXXXXXXX")
{
 go version;git rev-parse HEAD;git diff --stat
 printf 'cwd=%s\nrun=%s\nTMPDIR=%s\nGOCACHE=%s\nGOMODCACHE=%s\nmemprofilerate=%s\n' "$PWD" "$PROJECT_RUN_DIR" "$TMPDIR" "$GOCACHE" "$GOMODCACHE" "${PROFILE_MEM_RATE:-524288}"
 printf 'arguments:';printf ' %q' "${args[@]}" "${packages[@]}";printf '\n'
} > "$invocation/environment.txt"
go list "${list_args[@]}" -f '{{.ImportPath}} {{if or .TestGoFiles .XTestGoFiles}}tests{{else}}no-tests{{end}}' "${packages[@]}" > "$invocation/packages.txt"
status=0
while read -r package kind; do
 label="${package//\//-}";dir="$invocation/$label";mkdir "$dir"
 if [ "$kind" = no-tests ]; then printf '%s\n' 'No test files; no profile capture possible.' > "$dir/capture.txt";continue;fi
 binary="$PROJECT_RUN_DIR/${label}.test"
 set +e
 go test "${args[@]}" "$package" -cpuprofile="$dir/cpu.pprof" -memprofile="$dir/heap.pprof" -memprofilerate="${PROFILE_MEM_RATE:-524288}" -o "$binary" > "$dir/run.log" 2>&1
 code=$?
 set -e
 printf '%s\n' "$code" > "$dir/status.txt"
 # Execution has finished; retain the matching binary once, rather than
 # leaving a second disposable copy in every per-package run directory.
 [ ! -f "$binary" ] || mv "$binary" "$dir/package-test"
 cat "$dir/run.log"
 if [ -f "$dir/package-test" ] && [ -f "$dir/cpu.pprof" ] && [ -f "$dir/heap.pprof" ]; then
  go tool pprof -top -cum -nodecount=20 "$dir/package-test" "$dir/cpu.pprof" > "$dir/cpu-cumulative.txt"
  for sample in alloc_space alloc_objects inuse_space inuse_objects; do go tool pprof -sample_index="$sample" -top -cum -nodecount=20 "$dir/package-test" "$dir/heap.pprof" > "$dir/$sample.txt";done
  printf '%s\n' 'Profiles captured. Engineering hotspot/allocation conclusions required before acceptance. Fuzz/subprocess workers are not covered by parent profiles; inspect separately before accepting those workloads.' > "$dir/capture.txt"
 else
  printf '%s\n' 'Missing CPU/heap/binary capture (build failure/abrupt exit); profiling gate incomplete.' > "$dir/capture.txt"
 fi
 [ "$code" = 0 ] || status="$code"
done < "$invocation/packages.txt"
printf 'Retained test evidence: %s\n' "$invocation"
exit "$status"
