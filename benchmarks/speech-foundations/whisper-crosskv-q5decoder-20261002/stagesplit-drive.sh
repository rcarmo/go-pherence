#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
bash ./run-guarded.sh whisper-stagesplit-build go test -c -o $E/stagesplit.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in jfk pt pt2; do
  ENV_FILE=$E/stagesplit-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-stagesplit-$f $E/stagesplit.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "$f rc=$?"
done
rm -f $E/stagesplit.test
echo DRIVE_DONE
