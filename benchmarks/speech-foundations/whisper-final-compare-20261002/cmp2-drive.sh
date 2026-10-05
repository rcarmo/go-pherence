#!/usr/bin/env bash
set -u
E=/var/home/agent/workspace/tmp/whisper-vad-parity-20261001
cd $E
trap 'rm -f $E/cmp2-model.test; for c in $(podman ps -a --format "{{.Names}}" | grep "^whisper-cmp2"); do podman rm -f $c >/dev/null 2>&1; done' EXIT
bash ./run-guarded.sh whisper-cmp22-build go test -c -o $E/cmp2-model.test ./model/whisper || { echo BUILD_FAIL; exit 1; }
orig(){ RUN_GPU=1 bash ./run-guarded.sh whisper-cmp22-orig-$1 bun $E/run-original.ts $E/refresh-original-final /var/home/agent/workspace/projects/whisper-stt/models/ggml-large-v3-turbo-q5_0.bin $E/refresh-$2.f32 $3 /var/home/agent/workspace/projects/whisper-stt/models/ggml-silero-v6.2.0.bin $4 0 $E/cmp2-orig-$1.json; echo "orig $1 rc=$?"; }
gor(){ ENV_FILE=$E/cmp2-go-$1.env RUN_GPU=1 bash ./run-guarded.sh whisper-cmp22-go-$1 $E/cmp2-model.test -test.run '^TestWhisperPerformanceGoalArm$' -test.timeout 120s -test.v; echo "go $1 rc=$?"; }
# pass A: original first
orig jfk-a jfk en 0; gor jfk-a
orig pt-a pt pt 0; gor pt-a
orig pt2-a pt2 pt 0; gor pt2-a
orig jfkvad-a jfk en 1; gor jfkvad-a
orig groupsvad-a groups en 1; gor groupsvad-a
# pass B: Go first
gor jfk-b; orig jfk-b jfk en 0
gor pt-b; orig pt-b pt pt 0
gor pt2-b; orig pt2-b pt2 pt 0
gor jfkvad-b; orig jfkvad-b jfk en 1
gor groupsvad-b; orig groupsvad-b groups en 1
echo DRIVE_DONE
