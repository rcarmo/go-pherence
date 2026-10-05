#!/usr/bin/env bash
set -u
cd /var/home/agent/workspace/tmp/whisper-vad-parity-20261001
trap 'rm -f /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/mmqattr.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-mmqattr"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-mmqattr-build go test -c -o /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/mmqattr.test ./backends/vulkan || { echo BUILD_FAIL; exit 1; }
ENV_FILE=/var/home/agent/workspace/tmp/whisper-mmq-attrib-20261002/attrib.env RUN_GPU=1 bash ./run-guarded.sh whisper-mmqattr-run /var/home/agent/workspace/tmp/whisper-vad-parity-20261001/mmqattr.test -test.run '^TestZZMMQAttributionDiagnostic$' -test.timeout 120s -test.v
echo "run rc=$?"
echo DRIVE_DONE
