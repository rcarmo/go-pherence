#!/usr/bin/env bash
# Isolated sequential Whisper qualification; stops only its own container.
set -u
NAME=$1; shift
ROOT=/var/home/agent/workspace
EVIDENCE=$ROOT/tmp/whisper-vad-parity-20261001
check_idle() {
 local slots available
 available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
 [[ "$available" -ge 6291456 ]] || { echo 'host memory below6GiB'; return 1; }
 slots=$(curl -fsS --connect-timeout 2 --max-time 4 http://192.168.1.70:11434/slots 2>/dev/null) || { echo 'Qwen guard unavailable'; return 1; }
 printf '%s' "$slots" | jq -e 'type=="array" and length>0 and all(.[]; .is_processing == false)' >/dev/null || { echo 'Qwen not confirmed idle'; return 1; }
}
reason=$(check_idle) || { echo "$reason"; exit 75; }
mkdir -p "$EVIDENCE/cache" "$EVIDENCE/modcache"
extra=()
[[ ${RUN_GPU:-0} == 1 ]] && extra+=(--device /dev/dri/renderD128)
[[ -n ${ENV_FILE:-} ]] && extra+=(--env-file "$ENV_FILE")
podman run --name "$NAME" --network none --security-opt label=disable "${extra[@]}" --read-only --tmpfs /tmp:rw,size=2g --cpus 4 --memory 8g --memory-swap 8g --userns keep-id --user "$(id -u):$(id -g)" -v "$ROOT:$ROOT:ro" -v /opt/piclaw/current/bun/bin:/opt/piclaw/bun:ro -v /usr/bin/ps:/usr/bin/ps:ro -v /usr/lib64/libproc2.so.1.0.1:/lib64/libproc2.so.1:ro -v "$EVIDENCE:$EVIDENCE:rw" -v /var/home/agent/go/pkg/mod:/var/home/agent/go/pkg/mod:ro -w "$ROOT/projects/go-pherence-transcribe-web" -e HOME=/tmp -e PATH="$ROOT/tools/go-toolchain/go/bin:/opt/piclaw/bun:/usr/sbin:/usr/bin:/sbin:/bin" -e GOCACHE="$EVIDENCE/cache" -e GOPATH=/var/home/agent/go -e GOFLAGS=-p=2 -e GOTOOLCHAIN=local -e GOPROXY=off -e GOMAXPROCS=4 -e GO_PHERENCE_DISABLE_NVIDIA=1 -e GO_PHERENCE_VULKAN_CPU=0 -e GO_PHERENCE_VULKAN_ALLOW_CPU=0 -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json -e GO_PHERENCE_VULKAN_DEVICE=Intel -e WHISPER_THREADS=4 -e MESA_SHADER_CACHE_DIR=/tmp/mesa -e LD_LIBRARY_PATH="$ROOT/projects/whisper-stt/runtime/build-vulkan/bin" 73955bdf70a7b14e89100610502f01d2e4dc231697fd17e785275f538cad6204 taskset -c 0-7 "$@" > "$EVIDENCE/$NAME.log" 2>&1 &
runner=$!
reason=''
while kill -0 "$runner" 2>/dev/null; do
 reason=$(check_idle) || break
 if podman inspect "$NAME" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
  podman stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$NAME" >> "$EVIDENCE/$NAME-monitor.log" 2>/dev/null || true
 fi
 sleep 3
done
if [[ -n "$reason" ]]; then
 printf '%s GUARD_ABORT %s\n' "$(date -u +%FT%TZ)" "$reason" | tee -a "$EVIDENCE/$NAME-monitor.log"
 podman stop --time 20 "$NAME" >/dev/null 2>&1 || true
fi
wait "$runner"; result=$?
podman inspect "$NAME" --format '{{json .State}}' > "$EVIDENCE/$NAME-state.json" 2>/dev/null || true
podman rm "$NAME" >/dev/null 2>&1 || true
printf 'runner_exit=%s\nguard_reason=%s\n' "$result" "$reason" > "$EVIDENCE/$NAME-exit.txt"
[[ -n "$reason" ]] && exit 75
exit "$result"
