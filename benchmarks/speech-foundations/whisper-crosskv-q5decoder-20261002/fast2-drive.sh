#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
trap 'rm -f $E/fast2-model.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-fast2"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
cd $E
bash ./run-guarded.sh whisper-fast2-build go test -c -o $E/fast2-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  for arm in fast2; do
    ENV_FILE=$E/$arm-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-$arm-$f $E/fast2-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
    echo "$arm $f rc=$?"
  done
done

rm -f $E/fast2-model.test
echo DRIVE_DONE
