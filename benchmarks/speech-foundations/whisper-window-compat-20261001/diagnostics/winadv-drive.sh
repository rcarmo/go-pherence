#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
bash ./run-guarded.sh whisper-winadv-build go test -c -o $E/winadv-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  for arm in winadv advdef; do
    ENV_FILE=$E/$arm-candidate-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-$arm-$f $E/winadv-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
    echo "$arm $f rc=$?"
  done
done
echo DRIVE_DONE
