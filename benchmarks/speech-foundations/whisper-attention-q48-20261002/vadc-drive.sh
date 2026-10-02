#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/vadc-model.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-vadc"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-vadc-build go test -c -o $E/vadc-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for n in vadc0-jfk vadc0-groups vadc1-jfk vadc1-groups; do
  ENV_FILE=$E/$n.env RUN_GPU=1 bash ./run-guarded.sh whisper-$n $E/vadc-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "$n rc=$?"
done
echo DRIVE_DONE
