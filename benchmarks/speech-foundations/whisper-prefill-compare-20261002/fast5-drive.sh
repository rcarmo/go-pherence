#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/fast5-model.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-fast5"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-fast5-build go test -c -o $E/fast5-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
orig(){ RUN_GPU=1 bash ./run-guarded.sh whisper-fast5-orig-$1 bun $E/run-original.ts $E/refresh-original-final /var/home/agent/workspace/projects/whisper-stt/models/ggml-large-v3-turbo-q5_0.bin $E/refresh-pt2.f32 pt /var/home/agent/workspace/projects/whisper-stt/models/ggml-silero-v6.2.0.bin 0 0 $E/fast5-orig-$1.json; echo "orig $1 rc=$?"; }
orig pt2-c
for f in jfk pt fr jfk-vadwords groups pt2 podcast podcast-vadwords pt1 silence; do
  ENV_FILE=$E/fast5-$f.env RUN_GPU=1 bash ./run-guarded.sh whisper-fast5-$f $E/fast5-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v
  echo "fast5 $f rc=$?"
done
orig pt2-d
echo DRIVE_DONE
