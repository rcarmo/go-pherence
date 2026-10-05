#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/fast4-model.test $E/fast4-vk.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-fast4"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-fast4-buildvk go test -c -o $E/fast4-vk.test ./backends/vulkan || { echo BUILD_FAIL; exit 1; }
ENV_FILE=$E/padext.env RUN_GPU=1 bash ./run-guarded.sh whisper-fast4-padext $E/fast4-vk.test -test.run '^TestVulkanNativePaddedExtent$' -test.timeout 120s -test.v
echo "padext rc=$?"
bash ./run-guarded.sh whisper-fast4-build go test -c -o $E/fast4-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  ENV_FILE=$E/fast4-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-fast4-$f $E/fast4-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "fast4 $f rc=$?"
done
echo DRIVE_DONE
