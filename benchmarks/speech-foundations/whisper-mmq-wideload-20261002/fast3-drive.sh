#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/fast3-model.test $E/fast3-vk.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-fast3"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-fast3-buildvk go test -c -o $E/fast3-vk.test ./backends/vulkan || { echo BUILD_FAIL; exit 1; }
ENV_FILE=$E/mmqexact.env RUN_GPU=1 bash ./run-guarded.sh whisper-fast3-mmqexact $E/fast3-vk.test -test.run '^TestVulkanNativeQ5IntegerDotMMQExact$' -test.timeout 120s -test.v
echo "mmqexact rc=$?"
bash ./run-guarded.sh whisper-fast3-build go test -c -o $E/fast3-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  ENV_FILE=$E/fast3-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-fast3-$f $E/fast3-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "fast3 $f rc=$?"
done
echo DRIVE_DONE
