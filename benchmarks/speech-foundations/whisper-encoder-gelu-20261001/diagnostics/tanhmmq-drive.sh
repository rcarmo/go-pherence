#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
bash ./run-guarded.sh whisper-tanhmmq-build go test -c -o $E/tanhmmq-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
bash ./run-guarded.sh whisper-tanhmmq-vkbuild go test -c -o $E/tanhmmq-vk.test ./backends/vulkan || { echo VKBUILD_FAIL; exit 1; }
ENV_FILE=$E/gelu-native.env RUN_GPU=1 bash ./run-guarded.sh whisper-tanhmmq-gelu-native $E/tanhmmq-vk.test -test.run '^TestVulkanNativeGELUOriginalTanh$' -test.timeout 120s -test.v
echo "gelu-native rc=$?"
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  ENV_FILE=$E/tanhmmq-candidate-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-tanhmmq-$f $E/tanhmmq-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "$f rc=$?"
done
echo DRIVE_DONE
