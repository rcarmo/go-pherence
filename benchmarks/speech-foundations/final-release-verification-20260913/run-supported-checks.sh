#!/usr/bin/env bash
set -euo pipefail
out="$PWD/tmp/final-release-08213a4"
rm -rf "$out"
mkdir -p "$out"
: > "$out/status.tsv"
run_target() {
  local target=$1 start rc elapsed
  shift
  printf '=== %s ===\n' "$target" | tee "$out/$target.log"
  start=$(date +%s)
  set +e
  ( set -x; "$@" ) >>"$out/$target.log" 2>&1
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - start ))
  printf '%s\t%d\t%d\n' "$target" "$rc" "$elapsed" | tee -a "$out/status.tsv"
  if (( rc != 0 )); then tail -100 "$out/$target.log"; exit "$rc"; fi
}
run_script() {
  local target=$1 script=$2 start rc elapsed
  printf '=== %s ===\n' "$target" | tee "$out/$target.log"
  start=$(date +%s)
  set +e
  bash -o pipefail -c "set -eux; $script" >>"$out/$target.log" 2>&1
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - start ))
  printf '%s\t%d\t%d\n' "$target" "$rc" "$elapsed" | tee -a "$out/status.tsv"
  if (( rc != 0 )); then tail -100 "$out/$target.log"; exit "$rc"; fi
}
run_script speech-foundations-check 'go test -p=1 -count=1 -timeout=60s ./loader/audio ./loader/audio/media ./loader/numpy; go test -p=1 -count=1 -timeout=60s ./models/whisper -run "Test(ExactFrontend|ComputeMelFlatWithT|WindowPlan|CheckedTimestamp|PCMTranscribe|PCMVulkan|PCMDigitalSilence|SpeechFixture|TurboFixtureSelection|CheckedLoad|LoadEncoderSource|CheckedConfig|SpeechContext)"; go test -p=1 -count=1 -timeout=60s ./models/speaker/community1; go vet -p=1 ./loader/audio ./loader/audio/media ./loader/numpy ./models/whisper ./models/speaker/community1'
run_script speech-vulkan-offline-check 'go test -p=1 -count=1 -timeout=60s ./backends/vulkan -run "^(TestVulkanOffline|TestVulkanDispatchRejects|TestVkBuf|TestVkKernelCreate|TestVkHelpers|TestVkWrappers|TestLoadSPIRV)"'
run_script speech-vulkan-community-check 'go test -p=1 -count=1 -timeout=60s ./backends/vulkan -run "^(TestVulkanOffline(ChannelAffine|Conv2DCHW|LSTMCell|LSTMSequence)|TestVulkanOfflineShaderContractEmbedded)"; go test -p=1 -count=1 -timeout=60s ./models/speaker/community1 -run "^TestVulkan(BasicBlock|ResNetTrunk|Embedding|LSTM|Segmentation|Diarization)"; bun test scripts/check-vulkan-shaders.test.ts'
run_script speech-vulkan-community-server-check 'GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./runtime/speechjob -run "^TestVulkanCommunity"; GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjobserve -run "^(TestCommunityVulkan|TestCommunityPrepare|TestCombined|TestBuiltProfiles)"; go vet ./models/speaker/community1 ./runtime/speechjob ./cmd/audio/speechjobserve'
run_script speech-vulkan-encoder-check 'go test -p=1 -count=1 -timeout=60s ./models/whisper -run "^Test(VulkanEncoder(Layout|RejectsBeforeDevice|Ownership|PlanConstructorAdmission)|PCMVulkan.*)$"'
run_script speech-sincnet-fma-check 'go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run "^TestFMAColumns"; GODEBUG=cpu.avx2=off,cpu.fma=off go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run "^TestFMAColumns(Order|Invalid)"; go test -p=1 -count=1 -timeout=60s ./models/speaker/community1 -run "^TestSincNetLowered"; GODEBUG=cpu.avx2=off,cpu.fma=off go test -p=1 -count=1 -timeout=60s ./models/speaker/community1 -run "^TestSincNetLowered"'
run_script speech-community-gemm-check 'go test -p=1 -count=1 -timeout=90s ./backends/simd/runtime -run "^TestFMAMatrix"; GODEBUG=cpu.avx2=off,cpu.fma=off go test -p=1 -count=1 -timeout=90s ./backends/simd/runtime -run "^TestFMAMatrix(Exact|Rejects)"; go test -p=1 -count=1 -timeout=90s ./models/speaker/community1 -run "^Test(WeSpeakerTiled(Convolution|FullDepthOracle)|ExperimentalEmbeddingGEMM)"; GODEBUG=cpu.avx2=off,cpu.fma=off go test -p=1 -count=1 -timeout=90s ./models/speaker/community1 -run "^Test(WeSpeakerTiled(Convolution|FullDepthOracle)|ExperimentalEmbeddingGEMM)"'
run_script speech-job-check 'go test -p=1 -count=1 -timeout=90s ./runtime/resourcebudget ./runtime/speechjob ./runtime/speechjob/httpapi; go test -p=1 -count=1 -timeout=90s ./models/whisper -run "^TestPCM(Resume|Transcribe)"; go vet ./runtime/resourcebudget ./runtime/speechjob ./runtime/speechjob/httpapi ./models/whisper'
run_script speech-job-http-check 'go test -p=1 -count=1 -timeout=90s ./runtime/speechjob/httpapi; go test -p=1 -count=1 -timeout=90s ./runtime/speechjob -run "^TestListPage"; go vet ./runtime/speechjob/httpapi ./runtime/speechjob'
run_script speech-job-cli-check 'go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjob; go vet ./cmd/audio/speechjob'
run_script speech-job-serve-check 'GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjobserve; go vet ./cmd/audio/speechjobserve'
run_script speech-media-integration 'GO_PHERENCE_TEST_FFMPEG=1 go test -p=1 -count=1 -timeout=30s ./loader/audio/media -run TestFFmpegIntegration'
run_script speech-quality-freeze-check 'bun scripts/check-speech-quality-freeze.ts'
run_script speech-community-corpus-contract-check 'python3 -m unittest scripts/test_score_community1_corpus.py scripts/test_community1_pipeline_reference.py; python3 scripts/score_community1_corpus.py --manifest benchmarks/speech-foundations/community1-corpus-manifest.json --validate-only'
cat "$out/status.tsv"
