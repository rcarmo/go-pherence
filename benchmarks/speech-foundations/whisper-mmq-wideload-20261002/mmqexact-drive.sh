#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/mmqx.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-mmqx"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-mmqx-build go test -c -o $E/mmqx.test ./backends/vulkan || { echo BUILD_FAIL; exit 1; }
ENV_FILE=$E/mmqexact.env RUN_GPU=1 bash ./run-guarded.sh whisper-mmqx-run $E/mmqx.test -test.run '^TestVulkanNativeQ5IntegerDotMMQExact$' -test.timeout 120s -test.v
echo "mmqexact rc=$?"
echo DRIVE_DONE
