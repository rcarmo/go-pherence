#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
trap 'rm -f $E/xkv-model.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-xkv"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
cd $E
bash ./run-guarded.sh whisper-xkv-build go test -c -o $E/xkv-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
for f in fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  for arm in xkvbase xkv; do
    ENV_FILE=$E/$arm-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-$arm-$f $E/xkv-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
    echo "$arm $f rc=$?"
  done
done

rm -f $E/xkv-model.test
echo DRIVE_DONE
