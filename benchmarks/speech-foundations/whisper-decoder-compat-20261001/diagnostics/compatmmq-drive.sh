#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
bash ./run-guarded.sh whisper-compatmmq-build go test -c -o $E/compatmmq-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in pt jfk fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  ENV_FILE=$E/compatmmq-candidate-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-compatmmq-$f $E/compatmmq-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "$f rc=$?"
done
echo DRIVE_DONE
