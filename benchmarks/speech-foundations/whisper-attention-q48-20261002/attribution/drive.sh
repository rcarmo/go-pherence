#!/usr/bin/env bash
set -u
cd /var/home/agent/workspace/tmp/whisper-vad-parity-20261001
trap 'rm -f /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/attnattr.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-attnattr"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-attnattr-build go test -c -o /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/attnattr.test ./backends/vulkan || { echo BUILD_FAIL; exit 1; }
ENV_FILE=/var/home/agent/workspace/tmp/whisper-attn-attrib-20261002/attn.env RUN_GPU=1 bash ./run-guarded.sh whisper-attnattr-run /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/attnattr.test -test.run '^TestZZAttentionAttributionDiagnostic$' -test.timeout 120s -test.v
echo "run rc=$?"
echo DRIVE_DONE
