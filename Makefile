TMPDIR ?= $(if $(wildcard /workspace/tmp),/workspace/tmp,/tmp)
GOTMPDIR ?= $(TMPDIR)
PYTHON ?= python3
# MODELS_DIR remains an input alias for existing automation.
CHECKPOINTS_DIR ?= $(if $(MODELS_DIR),$(MODELS_DIR),checkpoints)
MODEL ?=
MODEL_DOWNLOAD_FLAGS ?=
MINICPMV_MODEL ?=
MINICPMV_SAFETENSORS ?=
MINICPMV_IMAGE ?=
MINICPMV_AUDIO_DURATION_MS ?=
MINICPMV_FLAGS ?=
export TMPDIR GOTMPDIR

SPACEMIT_PACKAGES := ./backends/spacemit/... ./cmd/spacemit/...

.PHONY: host-build host-vet host-test host-check spacemit-hardware-test model-layout-check

.PHONY: all build test docs-check docs-diagrams docs-diagrams-check test-cpu spacemit-host-check spacemit-cross-compile test-model-coverage gemma4-mtp-parity gemma4-mtp-strict-parity gemma4-mtp-native-parity gemma4-gpu-cpu-parity whisper-turbo-parity whisper-simd-parity whisper-cuda-parity whisper-gpu-graph-parity whisper-turbo-check whisper-backend-compare whisper-backend-podcast-compare whisper-a100-compare whisper-a100-podcast-compare whisper-int8-compare whisper-int8-podcast-compare model-coverage-tmpdir model-coverage model-coverage-json model-coverage-markdown model-coverage-csv model-coverage-snapshot model-coverage-snapshot-file model-coverage-snapshot-check model-coverage-runtime-roadmap model-coverage-runtime-roadmap-json model-coverage-next-runtime model-coverage-next-runtime-json model-coverage-pending model-coverage-references-pending model-coverage-runtime-pending model-coverage-execution-pending model-coverage-parity-pending model-coverage-readiness-pending model-coverage-references-gate model-coverage-runtime-gate model-coverage-execution-gate model-coverage-parity-gate model-coverage-readiness-gate clean server chat gen vet models-list models-download models-download-small models-download-qwen models-download-qwen3tts models-download-lfm2 models-download-minicpmv models-download-minicpmo models-download-gemma4 models-download-speaker models-download-one minicpmv-inspect minicpmv-version minicpmv-support-summary minicpmv-capabilities minicpmv-pending-runtime minicpmv-coverage-pending minicpmv-assets-check minicpmv-fixture-path minicpmv-fixture-summary minicpmv-fixture-ready minicpmv-inspect-model minicpmv-fixture-check minicpmv-check gguf-inspect gguf-smoke gguf-bench gguf-turboquant-smoke gguf-validate gguf-check gguf-ci gguf-inspect-qwen36-reap gguf-smoke-qwen36-reap gguf-validate-qwen36-reap gguf-bench-qwen36-reap gguf-check-qwen36-reap gguf-ci-qwen36-reap qwen3tts-inspect qwen3tts-fixture-coverage lfm2-inspect lfm2-fixture-coverage hunyuan3d-fixture-env hunyuan3d-inventory hunyuan3d-inspect hunyuan3d-image-fixture hunyuan3d-conditioner-fixture hunyuan3d-denoiser-fixture hunyuan3d-lowstep-fixture hunyuan3d-mesh-fixture trellis2-fixture-env trellis2-inventory trellis2-lowstep-fixture trellis2-ovoxel-inspect whisper whisper-k3 speaker-weights

all: build

# Optional MIT go-264 media backend. No default switch or model execution.
.PHONY: speech-go264-check speech-go264-public-check speech-go264-paired-check
speech-go264-check:
	CGO_ENABLED=0 GO_PHERENCE_DISABLE_NVIDIA=1 go test -mod=readonly -p=1 -count=1 -timeout=60s ./loader/audio/media
	go vet -mod=readonly -p=1 ./loader/audio/media

speech-go264-public-check:
	CGO_ENABLED=0 GO_PHERENCE_TEST_GO264_PUBLIC=1 go test -mod=readonly -p=1 -count=1 -timeout=60s ./loader/audio/media -run TestGo264PublicMedia

# Explicit CPU model work; requires host admission and pinned model/PCM paths.
speech-go264-paired-check:
	CGO_ENABLED=0 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_GO264_SPEECH=1 go test -mod=readonly -p=1 -count=1 -timeout=120s ./model/whisper -run TestGo264PairedSpeech

# Focused speech-foundation checks: no model weights, driver initialisation,
# service startup or performance benchmark. FFmpeg integration is opt-in below.
.PHONY: speech-foundations-check speech-media-integration speech-affine-check speech-quality-freeze-check speech-vulkan-offline-check speech-vulkan-static-check speech-vulkan-community-check speech-vulkan-community-server-check

# Cross-check pinned Whisper/Community model, policy, PLDA and provisional
# quality metadata against their source manifests. No assets/models are opened.
speech-quality-freeze-check:
	bun scripts/check-speech-quality-freeze.ts

# Mock-only Vulkan ABI/lifetime checks: never calls VulkanInit or opens a GPU.
speech-vulkan-offline-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/vulkan -run '^(TestVulkanOffline|TestVulkanDispatchRejects|TestVkBuf|TestVkKernelCreate|TestVkHelpers|TestVkWrappers|TestLoadSPIRV)'

# Model-free Community-1 operator foundations: checked CHW 3x3/1x1 convolution
# and channel-major prepared BatchNorm affine+ReLU. No device/model execution.
speech-vulkan-community-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/vulkan -run '^(TestVulkanOffline(ChannelAffine|Conv2DCHW|LSTMCell|LSTMSequence)|TestVulkanOfflineShaderContractEmbedded)'
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/speaker/community1 -run '^TestVulkan(BasicBlock|ResNetTrunk|Embedding|LSTM|Segmentation|Diarization)'
	bun test scripts/check-vulkan-shaders.test.ts

# Model-free Community-1 hybrid server lifecycle and configuration. No VulkanInit.
speech-vulkan-community-server-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./runtime/speechjob -run '^TestVulkanCommunity'
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjobserve -run '^(TestCommunityVulkan|TestCommunityPrepare|TestCombined|TestBuiltProfiles)'
	go vet ./model/speaker/community1 ./runtime/speechjob ./cmd/audio/speechjobserve

# Explicit real-GPU qualification in a coordinated compute window. No models
# or services; synthetic numerical fixtures and optional warm host-wall timing.
.PHONY: speech-vulkan-native-check speech-vulkan-community-native-check speech-vulkan-community-trained-check speech-vulkan-community-trained-recovery-check
speech-vulkan-native-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_SPEECH=1 go test -p=1 -count=1 -timeout=120s ./backends/vulkan -run '^TestVulkanNativeSpeech$$' -v

# Synthetic Community-1 block/trunk/embedding/LSTM/PCM parity. Requires an
# explicitly named hardware device and an authorised isolated compute window.
speech-vulkan-community-native-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_COMMUNITY=1 go test -p=1 -count=1 -timeout=300s ./model/speaker/community1 -run '^TestVulkanCommunityNative$$' -v

# One pinned 30-second trained sample through the fixed-window hybrid owner.
# Asset variables and the physical device name are mandatory; no downloads.
speech-vulkan-community-trained-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_PLDA_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_PLDA_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_PUBLIC_WAV)" || (echo 'Set GO_PHERENCE_COMMUNITY1_PUBLIC_WAV'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_COMMUNITY_DIARIZATION=1 go test -p=1 -count=1 -timeout=600s ./model/speaker/community1 -run '^TestVulkanCommunity1TrainedDiarization$$' -v

# Kill a trained native owner after one window, then complete the full sample in
# a fresh process. Uses the same mandatory assets/device variables as above.
speech-vulkan-community-trained-recovery-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_PLDA_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_PLDA_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_PUBLIC_WAV)" || (echo 'Set GO_PHERENCE_COMMUNITY1_PUBLIC_WAV'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_COMMUNITY_RECOVERY=1 go test -p=1 -count=1 -timeout=120s ./model/speaker/community1 -run '^TestVulkanCommunity1TrainedProcessRecovery$$' -v

# Whole resident encoder graph: offline contracts or authorised synthetic GPU test.
.PHONY: speech-vulkan-encoder-check speech-vulkan-encoder-native-check
speech-vulkan-encoder-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/whisper -run '^Test(VulkanEncoder(Layout|RejectsBeforeDevice|Ownership|PlanConstructorAdmission)|PCMVulkan.*)$$'

speech-vulkan-encoder-native-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_ENCODER=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestVulkanEncoderNative$$' -v

# Pinned trained Tiny, no implicit downloads. Optional FULL_TINY gate is in test.
.PHONY: speech-vulkan-trained-tiny-check
speech-vulkan-trained-tiny-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TRAINED=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestVulkanEncoderTrainedTiny$$' -v

# Public JFK speech fixture via temporary FFmpeg and pinned trained Tiny.
.PHONY: speech-vulkan-public-speech-check
speech-vulkan-public-speech-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_SPEECH_QUALITY=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestVulkanPCMSpeechTiny$$' -v

# Public PT/FR quality diagnostics + digital silence/multi-window parity.
.PHONY: speech-vulkan-corpus-check
speech-vulkan-corpus-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_CORPUS=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestVulkanPublicCorpus$$' -v

# Reference-only public fixture export; no foreign neural runtime in production.
.PHONY: speech-oracle-export-check
speech-oracle-export-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_ORACLE_EXPORT=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestSpeechOracleExport$$' -v

# Heavier pinned Turbo F16-checkpoint/F32-inference gate. Public speech has a
# separate opt-in; no downloads or automatic service lifecycle changes.
.PHONY: speech-vulkan-turbo-check
speech-vulkan-turbo-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO=1 go test -p=1 -count=1 -timeout=300s ./model/whisper -run '^TestVulkanTurbo$$' -v

# Attribution only: full Turbo layer-plans vs same stages separately fenced.
.PHONY: speech-vulkan-turbo-profile
speech-vulkan-turbo-profile:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO_PROFILE=1 go test -p=1 -count=1 -timeout=300s ./model/whisper -run '^TestVulkanTurboProfile$$' -v

# Explicit candidate kernels; neither target changes encoder defaults.
.PHONY: speech-vulkan-linear-regtile-check speech-vulkan-linear-f16-weight-check speech-vulkan-linear-q8-weight-check speech-vulkan-whisper-q8-weight-check speech-vulkan-turbo-q8-weight-check speech-vulkan-turbo-q8-robustness-check speech-vulkan-turbo-q8-podcast-check speech-vulkan-turbo-regtile-check
speech-vulkan-linear-regtile-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_LINEAR_REGTILE=1 go test -p=1 -count=1 -timeout=120s ./backends/vulkan -run '^TestVulkanNativeLinearRegTile$$' -v

# Packed IEEE-F16 weights with F32 activation/accumulation/output. No optional
# 16-bit Vulkan feature and no model/default integration. TIMING is optional.
speech-vulkan-linear-f16-weight-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_LINEAR_F16_WEIGHT=1 go test -p=1 -count=1 -timeout=120s ./backends/vulkan -run '^TestVulkanNativeLinearF16Weight$$' -v

# Per-output-row symmetric Q8 weights with F32 activation/accumulation/output.
# No optional 8-bit Vulkan feature and no model/default integration.
speech-vulkan-linear-q8-weight-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_LINEAR_Q8_WEIGHT=1 go test -p=1 -count=1 -timeout=120s ./backends/vulkan -run '^TestVulkanNativeLinearQ8Weight$$' -v

# Explicit trained Tiny projection-only Q8 quality/performance gate.
speech-vulkan-whisper-q8-weight-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_TINY_DIR)" || (echo 'Set GO_PHERENCE_WHISPER_TINY_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_WHISPER_Q8_WEIGHT=1 go test -p=1 -count=1 -timeout=120s ./model/whisper -run '^TestVulkanWhisperQ8Weight$$' -v

# Explicit trained Turbo Q8 short gate; set SPEECH=1 plus pinned JFK path for
# the full transcript/timing comparison. No serving/default selection.
speech-vulkan-turbo-q8-weight-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_TURBO_DIR)" || (echo 'Set GO_PHERENCE_WHISPER_TURBO_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO_Q8_WEIGHT=1 go test -p=1 -count=1 -timeout=180s ./model/whisper -run '^TestVulkanTurboQ8Weight$$' -v

# Full-Q8 versus exact-timestamp MLP-only Q8 on silence and 63s composition.
speech-vulkan-turbo-q8-robustness-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_TURBO_DIR)" || (echo 'Set GO_PHERENCE_WHISPER_TURBO_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_JFK_PATH)" || (echo 'Set GO_PHERENCE_WHISPER_JFK_PATH'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO_Q8_ROBUSTNESS=1 go test -p=1 -count=1 -timeout=180s ./model/whisper -run '^TestVulkanTurboQ8Robustness$$' -v

# Natural 90s/three-window parity at offset 300s in the pinned podcast. This is
# unlabeled robustness evidence, not WER acceptance or a serving default.
speech-vulkan-turbo-q8-podcast-check:
	@test -n "$(GO_PHERENCE_VULKAN_DEVICE)" || (echo 'Set GO_PHERENCE_VULKAN_DEVICE'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_TURBO_DIR)" || (echo 'Set GO_PHERENCE_WHISPER_TURBO_DIR'; exit 1)
	@test -n "$(GO_PHERENCE_WHISPER_PODCAST_PATH)" || (echo 'Set GO_PHERENCE_WHISPER_PODCAST_PATH'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO_Q8_PODCAST=1 go test -p=1 -count=1 -timeout=180s ./model/whisper -run '^TestVulkanTurboQ8Podcast$$' -v

speech-vulkan-turbo-regtile-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_VULKAN_TURBO_REGTILE=1 go test -p=1 -count=1 -timeout=300s ./model/whisper -run '^TestVulkanTurboRegTile$$' -v

# Optional offline validator/compiler qualification. Explicit new output path;
# missing tools fail (never skip). No Vulkan loader/device or model execution.
speech-vulkan-static-check:
	@test -n "$(VULKAN_SHADER_REPORT)" || (echo 'Set VULKAN_SHADER_REPORT to a new output directory'; exit 1)
	bun test scripts/check-vulkan-shaders.test.ts
	bun scripts/check-vulkan-shaders.ts --output "$(VULKAN_SHADER_REPORT)"

.PHONY: speech-sincnet-fma-check
speech-sincnet-fma-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run '^TestFMAColumns'
	GODEBUG=cpu.avx2=off,cpu.fma=off GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run '^TestFMAColumns(Order|Invalid)'
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/speaker/community1 -run '^TestSincNetLowered'
	GODEBUG=cpu.avx2=off,cpu.fma=off GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/speaker/community1 -run '^TestSincNetLowered'

# Trained CPU reference gates require a verified local conversion and explicit admission.
.PHONY: speech-community-segmentation-check speech-community-segmentation-strict
speech-community-segmentation-check:
	@test -n "$(GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_SEGMENTATION=1 go test -p=1 -count=1 -timeout=120s ./model/speaker/community1 -run '^TestCommunity1TrainedSegmentation$$' -v

# Known intermediate failures: this target must remain nonzero until corrected.
speech-community-segmentation-strict:
	@test -n "$(GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_SEGMENTATION=1 GO_PHERENCE_TEST_COMMUNITY1_STRICT=1 go test -p=1 -count=1 -timeout=120s ./model/speaker/community1 -run '^TestCommunity1TrainedSegmentation$$' -v

.PHONY: speech-community-embedding-check speech-community-embedding-strict
speech-community-embedding-check:
	@test -n "$(GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING=1 go test -p=1 -count=1 -timeout=180s ./model/speaker/community1 -run '^TestCommunity1TrainedEmbedding$$' -v

# Strict frontend/trunk/support failures remain visible; endpoint gate is always enforced.
speech-community-embedding-strict:
	@test -n "$(GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING=1 GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING_STRICT=1 go test -p=1 -count=1 -timeout=180s ./model/speaker/community1 -run '^TestCommunity1TrainedEmbedding$$' -v

.PHONY: speech-community-corpus-contract-check speech-community-diarization-check speech-community-diarization-lowest-ties speech-community-trained-corpus-check
# Model-free manifest/parser checks. Scoring saved results requires a separate
# pyannote.metrics environment and explicit hash-pinned local asset paths.
speech-community-corpus-contract-check:
	$(PYTHON) -m unittest scripts/test_prepare_ami_corpus.py scripts/test_score_speaker_words.py scripts/test_score_community1_quality.py scripts/test_score_community1_corpus.py scripts/test_community1_pipeline_reference.py
	$(PYTHON) scripts/score_community1_corpus.py --manifest benchmarks/speech-foundations/community1-corpus-manifest.json --validate-only

# Strict tie policy is the default; the pinned public sample currently rejects a tie.
speech-community-diarization-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_DIARIZATION=1 go test -p=1 -count=1 -timeout=180s ./model/speaker/community1 -run '^TestCommunity1TrainedDiarization$$' -v

# Explicit experimental deterministic tie policy, not NumPy tie identity parity.
speech-community-diarization-lowest-ties:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_DIARIZATION=1 GO_PHERENCE_DIARIZATION_LOWEST_TIES=1 go test -p=1 -count=1 -timeout=180s ./model/speaker/community1 -run '^TestCommunity1TrainedDiarization$$' -v

# Explicit bounded arbitrary-corpus CPU run. The WAV must already be canonical
# mono16k PCM and fit the experimental 128-window ceiling. No downloads.
speech-community-trained-corpus-check:
	@test -n "$(GO_PHERENCE_COMMUNITY1_CORPUS_WAV)" || (echo 'Set GO_PHERENCE_COMMUNITY1_CORPUS_WAV'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_CORPUS_WAV_SHA256)" || (echo 'Set GO_PHERENCE_COMMUNITY1_CORPUS_WAV_SHA256'; exit 1)
	@test -n "$(GO_PHERENCE_COMMUNITY1_CORPUS_SAMPLES)" || (echo 'Set GO_PHERENCE_COMMUNITY1_CORPUS_SAMPLES'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_CORPUS=1 go test -p=1 -count=1 -timeout=600s ./model/speaker/community1 -run '^TestCommunity1TrainedCorpus$$' -v

.PHONY: speech-community-gemm-check speech-community-gemm-timing
speech-community-gemm-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./backends/simd/runtime -run '^TestFMAMatrix'
	GODEBUG=cpu.avx2=off,cpu.fma=off GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./backends/simd/runtime -run '^TestFMAMatrix(Exact|Rejects)'
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./model/speaker/community1 -run '^Test(WeSpeakerTiled(Convolution|FullDepthOracle)|ExperimentalEmbeddingGEMM)'
	GODEBUG=cpu.avx2=off,cpu.fma=off GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./model/speaker/community1 -run '^Test(WeSpeakerTiled(Convolution|FullDepthOracle)|ExperimentalEmbeddingGEMM)'

speech-community-gemm-timing:
	@test -n "$(GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR)" || (echo 'Set GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_TEST_COMMUNITY1_GEMM_TIMING=1 go test -p=1 -count=1 -timeout=120s ./model/speaker/community1 -run '^TestWeSpeakerTiledTiming$$' -v

.PHONY: speech-job-check speech-job-http-check speech-job-cli-check speech-job-media-integration speech-job-serve-check speech-job-serve-integration speech-job-ui-check speech-job-trained-combined-check speech-job-trained-corpus-check
speech-job-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./runtime/resourcebudget ./runtime/speechjob ./runtime/speechjob/httpapi
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./model/whisper -run '^TestPCM(Resume|Transcribe)'
	go vet ./runtime/resourcebudget ./runtime/speechjob ./runtime/speechjob/httpapi ./model/whisper

speech-job-ui-check:
	@test -n "$(SPEECHJOB_BROWSER_OUT)" || (echo 'Set SPEECHJOB_BROWSER_OUT to a browser evidence directory'; exit 1)
	@mkdir -p "$(SPEECHJOB_BROWSER_OUT)"
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -c -o "$(SPEECHJOB_BROWSER_OUT)/fixture.test" ./runtime/speechjob/httpapi
	@trap 'rm -f "$(SPEECHJOB_BROWSER_OUT)/fixture.test"' EXIT; bun scripts/speechjob-ui-check.ts --binary "$(SPEECHJOB_BROWSER_OUT)/fixture.test" --out "$(SPEECHJOB_BROWSER_OUT)"

speech-job-serve-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjobserve
	go vet ./cmd/audio/speechjobserve

# Explicit trained CPU diagnostic: real seven-stage HTTP profile on the pinned
# public 30 s sample. Requires local Tiny/Community assets and prior host/model
# admission; does not qualify Tiny WER, LowestIndexTies or broad SA-WER/DER/JER.
speech-job-trained-combined-check:
	@test -n "$(SPEECHJOB_TRAINED_WHISPER_DIR)" || (echo 'Set SPEECHJOB_TRAINED_WHISPER_DIR'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_SEGMENTATION_DIR)" || (echo 'Set SPEECHJOB_TRAINED_SEGMENTATION_DIR'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_EMBEDDING_DIR)" || (echo 'Set SPEECHJOB_TRAINED_EMBEDDING_DIR'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_PLDA_DIR)" || (echo 'Set SPEECHJOB_TRAINED_PLDA_DIR'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_PUBLIC_WAV)" || (echo 'Set SPEECHJOB_TRAINED_PUBLIC_WAV'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_JFK_WAV)" || (echo 'Set SPEECHJOB_TRAINED_JFK_WAV'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 GO_PHERENCE_TEST_TRAINED_COMBINED=1 go test -p=1 -count=1 -timeout=360s ./cmd/audio/speechjobserve -run '^TestTrainedCombinedHTTPPublicSample$$' -v

# Hash-pinned arbitrary-corpus seven-stage HTTP diagnostic. Internal retained
# checkpoints may be exported for offline scoring; ambiguous speakers stay private.
speech-job-trained-corpus-check:
	@test -n "$(SPEECHJOB_TRAINED_CORPUS_WAV)" || (echo 'Set SPEECHJOB_TRAINED_CORPUS_WAV'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_CORPUS_WAV_SHA256)" || (echo 'Set SPEECHJOB_TRAINED_CORPUS_WAV_SHA256'; exit 1)
	@test -n "$(SPEECHJOB_TRAINED_CORPUS_SAMPLES)" || (echo 'Set SPEECHJOB_TRAINED_CORPUS_SAMPLES'; exit 1)
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 GO_PHERENCE_TEST_TRAINED_COMBINED_CORPUS=1 go test -p=1 -count=1 -timeout=360s ./cmd/audio/speechjobserve -run '^TestTrainedCombinedHTTPCorpus$$' -v

speech-job-serve-integration:
	GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 GO_PHERENCE_TEST_FFMPEG=1 go test -p=1 -count=3 -timeout=90s ./cmd/audio/speechjobserve -run '^Test(ServingProfile|StartQueue)' -v

speech-job-cli-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./cmd/audio/speechjob
	go vet ./cmd/audio/speechjob

speech-job-http-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./runtime/speechjob/httpapi
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=90s ./runtime/speechjob -run '^TestListPage'
	go vet ./runtime/speechjob/httpapi ./runtime/speechjob

speech-job-media-integration:
	GO_PHERENCE_TEST_FFMPEG=1 go test -p=1 -count=3 -timeout=90s ./runtime/speechjob -run '^Test(Go264|FFmpeg)Job' -v

speech-affine-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run '^TestAffineF32'
	GODEBUG=cpu.avx2=off,cpu.fma=off GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./backends/simd/runtime -run '^TestAffineF32'
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/speaker/community1 -run '^TestSincNet(FMA32|Normalization)'

speech-foundations-check:
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./loader/audio ./loader/audio/media ./loader/numpy
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/whisper -run 'Test(ExactFrontend|ComputeMelFlatWithT|WindowPlan|CheckedTimestamp|PCMTranscribe|PCMVulkan|PCMDigitalSilence|SpeechFixture|TurboFixtureSelection|CheckedLoad|LoadEncoderSource|CheckedConfig|SpeechContext)'
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=60s ./model/speaker/community1
	go vet -p=1 ./loader/audio ./loader/audio/media ./loader/numpy ./model/whisper ./model/speaker/community1

speech-media-integration:
	GO_PHERENCE_TEST_FFMPEG=1 GO_PHERENCE_DISABLE_NVIDIA=1 go test -p=1 -count=1 -timeout=30s ./loader/audio/media -run TestFFmpegIntegration

model-layout-check:
	go test ./docs -run '^TestModelLayout' -count=1
	bun test scripts/model-layout.test.ts

docs-check: docs-diagrams-check model-layout-check
	bun test scripts/check-doc-links.test.ts
	bun run scripts/check-doc-links.ts
	go test ./docs -count=1

docs-diagrams:
	bun run scripts/render-architecture.ts --output docs/architecture.svg
	bun run scripts/render-test-matrix.ts --output docs/test-matrix.svg

docs-diagrams-check:
	@tmp=$$(mktemp -d); trap 'rm -rf $$tmp' EXIT; \
	bun run scripts/render-architecture.ts --output $$tmp/architecture.svg >/dev/null; \
	bun run scripts/render-test-matrix.ts --output $$tmp/test-matrix.svg >/dev/null; \
	cmp docs/architecture.svg $$tmp/architecture.svg; \
	cmp docs/test-matrix.svg $$tmp/test-matrix.svg

build: gen server chat

gen:
	go build -o bin/llmgen ./cmd/llm/llmgen

server:
	go build -o bin/llmserver ./cmd/llm/llmserver

chat:
	go build -o bin/llmchat ./cmd/llm/llmchat

minicpmv-inspect:
	go build -o bin/minicpmvinspect ./cmd/minicpmvinspect

minicpmv-version:
	go run ./cmd/minicpmvinspect -version $(MINICPMV_FLAGS)

minicpmv-support-summary:
	go run ./cmd/minicpmvinspect -support-summary $(MINICPMV_FLAGS)

minicpmv-capabilities:
	go run ./cmd/minicpmvinspect -capabilities $(MINICPMV_FLAGS)

minicpmv-pending-runtime:
	go run ./cmd/minicpmvinspect -pending-runtime-steps $(MINICPMV_FLAGS)

minicpmv-coverage-pending:
	$(MAKE) model-coverage-pending MODEL_COVERAGE_FAMILY=minicpmv

minicpmv-assets-check:
	$(PYTHON) scripts/minicpmv_assets_check.py --checkpoints-dir $(CHECKPOINTS_DIR) $(MINICPMV_FLAGS)

minicpmv-fixture-path:
	go run ./cmd/minicpmvinspect -fixture-path $(MINICPMV_FLAGS)

minicpmv-fixture-summary:
	go run ./cmd/minicpmvinspect -fixture-summary $(MINICPMV_FLAGS)

minicpmv-fixture-ready:
	go run ./cmd/minicpmvinspect -require-fixture-ready $(MINICPMV_FLAGS)

minicpmv-inspect-model:
	@if [ -z "$(MINICPMV_MODEL)" ]; then echo "usage: make minicpmv-inspect-model MINICPMV_MODEL=checkpoints/minicpm-v-2.6 [MINICPMV_SAFETENSORS=...] [MINICPMV_IMAGE=...] [MINICPMV_AUDIO_DURATION_MS=1234] [MINICPMV_FLAGS='-json']"; exit 2; fi
	go run ./cmd/minicpmvinspect -model $(MINICPMV_MODEL) $(if $(MINICPMV_SAFETENSORS),-safetensors $(MINICPMV_SAFETENSORS),) $(if $(MINICPMV_IMAGE),-image $(MINICPMV_IMAGE),) $(if $(MINICPMV_AUDIO_DURATION_MS),-audio-duration-ms $(MINICPMV_AUDIO_DURATION_MS),) $(MINICPMV_FLAGS)

minicpmv-fixture-check:
	go test ./model/minicpmv -run TestMiniCPMOFixtureMetadata -count=1
	go run ./cmd/minicpmvinspect -require-fixture-ready
	go run ./cmd/minicpmvinspect -model model/minicpmv/testdata/minicpmo_fixture -require-metadata-ready -audio-duration-ms 1234

minicpmv-check:
	$(PYTHON) scripts/minicpmv_check.py

# Whisper speech-to-text. The optimized RVV + SpaceMIT IME (int8) kernels are
# gated by //go:build riscv64 and selected at runtime via CPU feature detection,
# so a native riscv64 build picks them up automatically. See
# docs/performance/whisper-riscv-optimization.md for the optimization details and the
# WHISPER_* runtime tunables.
whisper:
	go build -o bin/whisper ./cmd/audio/whisper

.PHONY: moss-transcribe moss-transcribe-parity
moss-transcribe:
	go build -o bin/moss-transcribe ./cmd/audio/moss-transcribe

moss-transcribe-parity:
	@if [ -z "$(MOSS_TRANSCRIBE_MODEL_DIR)" ]; then echo "usage: make moss-transcribe-parity MOSS_TRANSCRIBE_MODEL_DIR=/path/to/MOSS-Transcribe-Diarize"; exit 2; fi
	MOSS_TRANSCRIBE_MODEL_DIR=$(MOSS_TRANSCRIBE_MODEL_DIR) go test ./model/mosstranscribe -run 'TestRealCheckpoint(AudioBackbone|QwenDecoderLoads|WhisperEncoderParity|NativeModelLoads|JFKTranscriptParity)|TestPinnedMOSSRealTokenizer' -count=1 -v

# Build for the SpaceMIT K1/K3 (MilkV Jupiter 2: 8x X60 RISC-V, RVV 1.0 + IME
# integer matrix engine). Forces GOARCH=riscv64 so it can be cross-compiled from
# an x86 host as well as built natively on the board; CGO disabled for a static
# binary. Run with WHISPER_INT8=1 for the full int8 IME pipeline.
whisper-k3:
	mkdir -p $(GOTMPDIR)
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/whisper-k3 ./cmd/audio/whisper
	@echo "Built bin/whisper-k3 (riscv64, RVV + IME int8)."
	@echo "Resident run:  WHISPER_INT8=1 WHISPER_THREADS=4 bin/whisper-k3 -model <model.safetensors> -size large-v3 -audio <file.wav>"

whisper-turbo-parity:
	mkdir -p $(GOTMPDIR)
	WHISPER_REQUIRE_TURBO_PARITY=1 GOTMPDIR=$(GOTMPDIR) go test ./model/whisper -run TestLargeV3TurboJFKCPUTranscriptParity -count=1

whisper-simd-parity:
	mkdir -p $(GOTMPDIR)
	GOTMPDIR=$(GOTMPDIR) go test ./model/whisper ./loader/audio ./backends/simd/fft ./backends/simd/runtime -run 'TestWhisperConv1DFastMatchesScalarOracle|TestWhisperLayerNormUsesSIMDOracleMatchesScalar|TestWhisperFullAttentionMatchesScalarOracle|TestLinearRowBlockUsesSIMDOracleMatchesScalar|TestMelSpectrogramMatchesReferencePath|TestMelSpectrogramFusedUsesLog10|TestDotI8F32|TestDotI8F32x4|TestSdotx4|TestQ4RowDot' -count=1 -v

whisper-cuda-parity:
	mkdir -p $(GOTMPDIR)
	GOTMPDIR=$(GOTMPDIR) go test ./model/whisper -run 'TestGPUEncoderForwardNotReadyFallbackMatchesCPU|TestWhisperCUDA|TestWhisperGPUGraphUmbrella|TestWhisperGPUFeatureFlags|TestNewDecoderStateGPU' -count=1 -v
	GOTMPDIR=$(GOTMPDIR) go test ./backends/nvidia/runtime -run TestWhisperAttentivePoolParity -count=1 -v

whisper-gpu-graph-parity:
	mkdir -p $(GOTMPDIR)
	WHISPER_REQUIRE_TURBO_PARITY=1 GO_PHERENCE_WHISPER_GPU_GRAPH=1 GOTMPDIR=$(GOTMPDIR) go test ./model/whisper -run TestLargeV3TurboJFKCPUTranscriptParity -count=1 -v

whisper-turbo-check: whisper-turbo-parity whisper-simd-parity whisper-cuda-parity whisper-gpu-graph-parity
	mkdir -p $(GOTMPDIR)
	GOTMPDIR=$(GOTMPDIR) go test ./model/whisper ./cmd/audio/... ./loader/audio ./backends/simd/fft ./backends/simd/runtime ./backends/cuda/ptx -count=1
	$(PYTHON) scripts/whisper_turbo_smoke.py --audio testdata/jfk.wav
	$(PYTHON) scripts/speakercheck_suite.py testdata/speakercheck_suite.json

whisper-backend-compare: whisper-a100-compare whisper-int8-compare

whisper-backend-podcast-compare: whisper-a100-podcast-compare whisper-int8-podcast-compare

whisper-a100-compare:
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 16
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 16 --task transcribe --language en
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 8 --timestamps
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 8 --timestamps --task transcribe --language en
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt --speaker-model $(SPEAKER_SAFETENSORS)
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt --task transcribe --language en --speaker-model $(SPEAKER_SAFETENSORS)

whisper-a100-podcast-compare:
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --timestamps
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --diarize-vtt
	$(PYTHON) scripts/whisper_a100_compare.py --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --diarize-vtt --speaker-model $(SPEAKER_SAFETENSORS)

whisper-int8-compare:
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 16
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 16 --task transcribe --language en
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 8 --timestamps
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 8 --timestamps --task transcribe --language en
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt --speaker-model $(SPEAKER_SAFETENSORS)
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/jfk.wav --max-tokens 8 --diarize-vtt --task transcribe --language en --speaker-model $(SPEAKER_SAFETENSORS)

whisper-int8-podcast-compare:
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --timestamps
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --diarize-vtt
	$(PYTHON) scripts/whisper_a100_compare.py --backend int8 --audio testdata/podcast.wav --start 300 --duration 12 --max-tokens 4 --diarize-vtt --speaker-model $(SPEAKER_SAFETENSORS)

.PHONY: ideogram4gen-k3 ideogram4-k3-check
ideogram4gen-k3:
	mkdir -p $(GOTMPDIR) bin
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/ideogram4gen-k3 ./cmd/image/ideogram4gen
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/ideogram4vaeprobe-k3 ./cmd/image/ideogram4vaeprobe
	@echo "Built bin/ideogram4gen-k3 and bin/ideogram4vaeprobe-k3 (riscv64 target)."
	@echo "Hardware smoke: IME2_TCM_ACT=1 bin/ideogram4gen-k3 -k3 -k3-threads 8 -k3-prewarm -model <ideogram4-dir> -prompt \"\$$(cat prompts/ideogram4/cat.json)\" -width 256 -height 256 -steps 4 -guidance 7.0 -mu 0.0 -std 1.75 -seed 2026060803 -timing"
	@echo "VAE smoke:      bin/ideogram4vaeprobe-k3 -k3 -k3-threads 8 -k3-prewarm -model <ideogram4-dir> -width 256 -height 256"

ideogram4-k3-check:
	mkdir -p $(GOTMPDIR) $(TMPDIR)/ideogram4/k3check bin
	GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/ideogram4 ./cmd/image/ideogram4gen ./backends/nvidia/runtime
	@for pkg in ./model/ideogram4 ./cmd/image/ideogram4gen ./cmd/image/ideogram4vaeprobe ./backends/spacemit/rvv ./backends/spacemit/ime2 ./backends/spacemit/inference ./backends/spacemit/k3engine/aipool ./backends/spacemit/k3engine; do \
		out="$(TMPDIR)/ideogram4/k3check/$$(echo $$pkg | sed 's#[/.]#_#g').test"; \
		echo "cross-test $$pkg -> $$out"; \
		CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go test -c -o "$$out" "$$pkg" || exit $$?; \
	done
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/ideogram4gen-k3 ./cmd/image/ideogram4gen
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/ideogram4vaeprobe-k3 ./cmd/image/ideogram4vaeprobe
	./scripts/ideogram4_k3_coverage.py --fail-missing > $(TMPDIR)/ideogram4/k3check/coverage.json
	@echo "K3 check passed: native Ideogram tests + riscv64 test binaries + bin/ideogram4gen-k3 + bin/ideogram4vaeprobe-k3 + coverage no-missing gate"

SPEAKER_CKPT ?= $(CHECKPOINTS_DIR)/speechbrain-ecapa-voxceleb/embedding_model.ckpt
SPEAKER_SAFETENSORS ?= $(CHECKPOINTS_DIR)/speaker-ecapa-voxceleb.safetensors
SPEAKER_CKPT_URL ?= https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt

# Download + convert the SpeechBrain ECAPA-TDNN speaker-embedding weights WITHOUT
# torch (works on the RISC-V board; needs only python3 + numpy). Produces the
# safetensors that `whisper -diarize` and cmd/audio/speakercheck consume. The full
# torch-based converter (scripts/convert_speechbrain_ecapa.py) remains for hosts
# that have torch installed.
speaker-weights:
	mkdir -p $(dir $(SPEAKER_CKPT))
	[ -f $(SPEAKER_CKPT) ] || curl -fsSL -o $(SPEAKER_CKPT) "$(SPEAKER_CKPT_URL)"
	$(PYTHON) scripts/ckpt_to_safetensors_numpy.py --checkpoint $(SPEAKER_CKPT) --output $(SPEAKER_SAFETENSORS)
	@echo "Wrote $(SPEAKER_SAFETENSORS). Diarize with: whisper -timestamps -diarize -audio <file.wav> ..."

test:
	go test -count=1 -timeout=120s ./loader/... ./model/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

# Native OmniVoice numerical core and audio frontend; no model download required.
.PHONY: test-omnivoice vet-omnivoice build-omnivoice
test-omnivoice:
	go test -count=1 -timeout=120s ./loader/tokenizer ./loader/omnivoice ./model/omnivoice ./cmd/audio/omnivoice

vet-omnivoice:
	go vet ./loader/tokenizer ./loader/omnivoice ./model/omnivoice ./cmd/audio/omnivoice

build-omnivoice:
	mkdir -p bin
	go build -o bin/omnivoice ./cmd/audio/omnivoice

test-cpu:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_VULKAN_ALLOW_CPU=0 go test -count=1 -timeout=120s ./loader/... ./model/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

# Respect host build constraints; never force foreign tags or omit backend folders.
host-build:
	go build ./...

host-vet:
	go vet ./...

host-test:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_VULKAN_ALLOW_CPU=0 go test -count=1 -timeout=120s ./...

host-check:
	$(MAKE) model-layout-check
	$(MAKE) host-build
	$(MAKE) host-vet
	$(MAKE) host-test

# Portable packing/fallback tests run on the host; AICPU and TCM need Linux/RISC-V.
spacemit-host-check:
	go build $(SPACEMIT_PACKAGES)
	go vet $(SPACEMIT_PACKAGES)
	GO_PHERENCE_DISABLE_NVIDIA=1 go test -count=1 -timeout=120s $(SPACEMIT_PACKAGES)

# Explicit board-only execution, separate from cross-compilation.
spacemit-hardware-test:
	@test "$$(go env GOOS)/$$(go env GOARCH)" = linux/riscv64 || (echo "Requires a Linux/RISC-V K3 board" >&2; exit 2)
	@test -e /proc/set_ai_thread || (echo "K3 AI-core registration is unavailable" >&2; exit 2)
	GO_PHERENCE_TEST_K3=1 go test -count=1 -timeout=120s $(SPACEMIT_PACKAGES)

# Compile only: never execute these RISC-V test binaries on the host.
spacemit-cross-compile:
	mkdir -p $(GOTMPDIR)/spacemit-riscv64
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build $(SPACEMIT_PACKAGES)
	@for pkg in ime2 inference aicpu aicpu/aipool; do \
		name=$$(echo $$pkg | tr / -); \
		CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go test -c \
			-o $(GOTMPDIR)/spacemit-riscv64/$$name.test ./backends/spacemit/$$pkg || exit $$?; \
	done

gemma4-mtp-parity:
	GOTMPDIR=$(GOTMPDIR) go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1
	GOTMPDIR=$(GOTMPDIR) go run ./cmd/models/gemma4mtpparity -fixture model/testdata/gemma4-mtp-llamacpp-fixture.json -pretty=false
	GOTMPDIR=$(GOTMPDIR) go test ./loader/gguf -run 'TestDequantRowQ4KToZeroBlock|TestDequantRowQ4KToMatchesGGMLNibbleGroups|TestExpertMatricesQ4KGemvMatchesDequantScalar|TestDequantRowQ8_0ToMatchesScaleTimesInt8|TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale|TestDotQ4_0Q8_0MatchesAVX2Reference|TestDotQ4_0Q8_0MatchesScalarReference|TestQuantizeQ8KComputesScaleQuantsAndBlockSums|TestDequantRowQ6KToMatchesScalarReference|TestDotQ6KQ8KMatchesAVX2Reference|TestDotQ6KQ8KMatchesScalarReference' -count=1

gemma4-mtp-strict-parity:
	@test -n "$(GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE)" || (echo "GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE is required for strict selected-logit parity" >&2; exit 2)
	GOTMPDIR=$(GOTMPDIR) go run ./cmd/models/gemma4mtpparity -fixture "$(GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE)" $(if $(GO_PHERENCE_GEMMA4_MAIN),-model "$(GO_PHERENCE_GEMMA4_MAIN)",) $(if $(GO_PHERENCE_GEMMA4_MTP_DRAFTER),-drafter "$(GO_PHERENCE_GEMMA4_MTP_DRAFTER)",)
	GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE="$(GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE)" $(if $(GO_PHERENCE_GEMMA4_MAIN),GO_PHERENCE_GEMMA4_MAIN="$(GO_PHERENCE_GEMMA4_MAIN)",) $(if $(GO_PHERENCE_GEMMA4_MTP_DRAFTER),GO_PHERENCE_GEMMA4_MTP_DRAFTER="$(GO_PHERENCE_GEMMA4_MTP_DRAFTER)",) GOTMPDIR=$(GOTMPDIR) go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1

gemma4-mtp-native-parity:
	@test -n "$(GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE)" || (echo "GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE is required for bounded native-Go parity" >&2; exit 2)
	GO_PHERENCE_GEMMA4_MTP_NATIVE_LOGIT_TOLERANCE=0.2 GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE="$(GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE)" $(if $(GO_PHERENCE_GEMMA4_MAIN),GO_PHERENCE_GEMMA4_MAIN="$(GO_PHERENCE_GEMMA4_MAIN)",) $(if $(GO_PHERENCE_GEMMA4_MTP_DRAFTER),GO_PHERENCE_GEMMA4_MTP_DRAFTER="$(GO_PHERENCE_GEMMA4_MTP_DRAFTER)",) GOTMPDIR=$(GOTMPDIR) go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1

gemma4-gpu-cpu-parity:
	flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GEMMA4_TRACE_TEST=1 GO_PHERENCE_GPU_KV_MAX_SEQ=64 GOTMPDIR=$(GOTMPDIR) go test -tags "diagnostic gemma4fixtures" ./model/gemma4 -run "^TestGemma4GPUGenerate$$" -count=1 -v'
	flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GEMMA4_TRACE_TEST=1 GO_PHERENCE_GPU_KV_MAX_SEQ=64 GOTMPDIR=$(GOTMPDIR) go test -tags "diagnostic gemma4fixtures" ./model/gemma4 -run "^TestGemma4QuantizedCPUvsGPULayerWalk$$" -count=1 -v'
	flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GEMMA4_TRACE_TEST=1 GO_PHERENCE_GPU_KV_MAX_SEQ=64 GOTMPDIR=$(GOTMPDIR) go test -tags "diagnostic gemma4fixtures" ./model/gemma4 -run "^TestGemma4CPUvsGPUProjectionTrace$$" -count=1 -v'
	flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GEMMA4_TRACE_TEST=1 GO_PHERENCE_GPU_KV_MAX_SEQ=64 GOTMPDIR=$(GOTMPDIR) go test -tags "diagnostic gemma4fixtures" ./model/gemma4 -run "^TestGemma4QuantizedCPUvsGPUOpTrace$$" -count=1 -v'
	flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GEMMA4_TRACE_TEST=1 GO_PHERENCE_GPU_KV_MAX_SEQ=64 GOTMPDIR=$(GOTMPDIR) go test -tags "diagnostic gemma4fixtures" ./model/gemma4 -run "^TestGemma4QuantizedCPUvsGPUOpTraceEarly$$" -count=1 -v'

test-model-coverage: model-coverage-tmpdir
	go test -count=1 -timeout=120s ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen/qwen3ttsinspect ./cmd/models/lfm2inspect ./cmd/models/modelcoverage
	go vet ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen/qwen3ttsinspect ./cmd/models/lfm2inspect ./cmd/models/modelcoverage
	go run ./cmd/models/modelcoverage -references-only -fail-pending
	go run ./cmd/models/modelcoverage -parity-only -fail-pending
	go run ./cmd/models/modelcoverage -readiness-only -fail-pending
	go run ./cmd/models/modelcoverage -min-percent $(MODEL_COVERAGE_MIN_PERCENT)
	$(MAKE) model-coverage-snapshot-check

MODEL_COVERAGE_FAMILY ?=
MODEL_COVERAGE_MIN_PERCENT ?= 90

model-coverage-tmpdir:
	mkdir -p $(GOTMPDIR)

model-coverage: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),)

model-coverage-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -json

model-coverage-markdown: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -markdown

model-coverage-csv: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -csv

model-coverage-snapshot: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -snapshot

model-coverage-snapshot-file: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -snapshot > docs/model-coverage-snapshot.md

model-coverage-snapshot-check: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage -snapshot > $(GOTMPDIR)/model-coverage-snapshot.check.md
	cmp docs/model-coverage-snapshot.md $(GOTMPDIR)/model-coverage-snapshot.check.md

model-coverage-runtime-roadmap: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap

model-coverage-runtime-roadmap-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap-json

model-coverage-next-runtime: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -next-runtime

model-coverage-next-runtime-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -next-runtime-json

model-coverage-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -pending-only

model-coverage-references-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -pending-only

model-coverage-runtime-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -pending-only

model-coverage-execution-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -execution-only -pending-only

model-coverage-parity-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -pending-only

model-coverage-readiness-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -pending-only

model-coverage-references-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -fail-pending

model-coverage-runtime-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -fail-pending

model-coverage-execution-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -execution-only -fail-pending

model-coverage-parity-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -fail-pending

model-coverage-readiness-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -fail-pending

vet:
	go vet ./...

clean:
	rm -rf bin/

models-list:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --dry-run $(MODEL_DOWNLOAD_FLAGS)

models-download:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) $(MODEL_DOWNLOAD_FLAGS)

models-download-small:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group small $(MODEL_DOWNLOAD_FLAGS)

models-download-qwen:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group qwen $(MODEL_DOWNLOAD_FLAGS)

models-download-qwen3tts:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group qwen3tts $(MODEL_DOWNLOAD_FLAGS)

models-download-lfm2:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group lfm2 $(MODEL_DOWNLOAD_FLAGS)

models-download-minicpmv:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group minicpmv --group minicpmo $(MODEL_DOWNLOAD_FLAGS)

models-download-minicpmo:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group minicpmo $(MODEL_DOWNLOAD_FLAGS)

models-download-gemma4:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group gemma4 $(MODEL_DOWNLOAD_FLAGS)

models-download-speaker:
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --group speaker $(MODEL_DOWNLOAD_FLAGS)

models-download-one:
	@if [ -z "$(MODEL)" ]; then echo "usage: make models-download-one MODEL=qwen3.6-27b-mlx4-mtp"; exit 2; fi
	$(PYTHON) scripts/download_models.py --checkpoints-dir $(CHECKPOINTS_DIR) --only $(MODEL) $(MODEL_DOWNLOAD_FLAGS)

GGUF_MODEL ?= /opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf
GGUF_PROMPT_IDS ?= 0
GGUF_MAX_NEW ?= 1
GGUF_CACHE_TYPE_K ?= turbo4
GGUF_CACHE_TYPE_V ?= turbo2
GGUF_KV_RESIDUAL_WINDOW ?= 128
GGUF_KV_SMOKE_TOKENS ?= 5
GGUF_EXPECT_KV_SMOKE_LAYER ?=
GGUF_EXPECT_KV_SMOKE_COMPRESSED ?=
GGUF_EXPECT_KV_SMOKE_FULL ?=
GGUF_EXPECT_KV_SMOKE_BYTES ?=
GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES ?=
GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES ?=
GGUF_EXPECT_GENERATED ?=
GGUF_EXPECT_DECODED ?=
GGUF_EXPECT_RUNTIME_FLOAT_BYTES ?=
GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES ?=
GGUF_EXPECT_RUNTIME_SCRATCH_BYTES ?=
GGUF_EXPECT_RUNTIME_TOTAL_BYTES ?=
GGUF_EXPECT_KV_COMPRESSED_LAYERS ?=
GGUF_EXPECT_KV_SEQ ?=
GGUF_EXPECT_KV_COMPRESSED_COUNT ?=
GGUF_EXPECT_KV_FULL_COUNT ?=
GGUF_EXPECT_KV_FLOAT_BYTES ?=
GGUF_EXPECT_KV_COMPRESSED_BYTES ?=
GGUF_EXPECT_KV_SCRATCH_BYTES ?=
GGUF_EXPECT_KV_TOTAL_BYTES ?=
GGUF_EXPECT_REAP_RATIO ?=
GGUF_EXPECT_REAP_SOURCE ?=
GGUF_EXPECT_ARCHITECTURE ?=
GGUF_EXPECT_NAME_CONTAINS ?=
GGUF_EXPECT_TENSOR_COUNT ?=
GGUF_EXPECT_LAYERS ?=
GGUF_EXPECT_HIDDEN_SIZE ?=
GGUF_EXPECT_HEADS ?=
GGUF_EXPECT_VOCAB_SIZE ?=
GGUF_EXPECT_TOKENIZER_TOKENS ?=
GGUF_EXPECT_BOS ?=
GGUF_EXPECT_EOS ?=
GGUF_EXPECT_MAX_SEQ_LEN ?=
GGUF_EXPECT_FULL_ATTENTION_INTERVAL ?=
GGUF_EXPECT_KV_HEADS ?=
GGUF_EXPECT_HEAD_DIM ?=
GGUF_EXPECT_KV_DIM ?=
GGUF_EXPECT_EXPERTS ?=
GGUF_EXPECT_EXPERTS_PER_TOKEN ?=
GGUF_EXPECT_F32_COUNT ?=
GGUF_EXPECT_Q4_K_COUNT ?=
GGUF_EXPECT_Q6_K_COUNT ?=
GGUF_EXPECT_CACHE_LAYERS ?=
GGUF_EXPECT_PROTECTED_CACHE_LAYERS ?=
GGUF_EXPECT_FULL_KV_BYTES ?=
GGUF_EXPECT_ESTIMATED_KV_BYTES ?=
GGUF_EXPECT_SAVED_KV_BYTES ?=
GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES ?=
GGUF_EXPECT_ESTIMATED_TOTAL_BYTES ?=
GGUF_EXPECT_SIMD_ROTATION ?=
GGUF_CI_PACKAGES ?= ./cmd/llm/llmserver ./loader/gguf ./cmd/models/ggufinspect ./cmd/models/ggufsmoke ./model ./runtime/kv

# Inspect and smoke the native pure-Go/SIMD GGUF path for llama/Qwen REAP models.
gguf-inspect:
	go run ./cmd/models/ggufinspect -json -require-runtime-ready -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_REAP_RATIO),-expect-reap-ratio $(GGUF_EXPECT_REAP_RATIO),) $(if $(GGUF_EXPECT_REAP_SOURCE),-expect-reap-source $(GGUF_EXPECT_REAP_SOURCE),) $(if $(GGUF_EXPECT_ARCHITECTURE),-expect-architecture $(GGUF_EXPECT_ARCHITECTURE),) $(if $(GGUF_EXPECT_NAME_CONTAINS),-expect-name-contains $(GGUF_EXPECT_NAME_CONTAINS),) $(if $(GGUF_EXPECT_TENSOR_COUNT),-expect-tensor-count $(GGUF_EXPECT_TENSOR_COUNT),) $(if $(GGUF_EXPECT_LAYERS),-expect-layers $(GGUF_EXPECT_LAYERS),) $(if $(GGUF_EXPECT_HIDDEN_SIZE),-expect-hidden-size $(GGUF_EXPECT_HIDDEN_SIZE),) $(if $(GGUF_EXPECT_HEADS),-expect-heads $(GGUF_EXPECT_HEADS),) $(if $(GGUF_EXPECT_VOCAB_SIZE),-expect-vocab-size $(GGUF_EXPECT_VOCAB_SIZE),) $(if $(GGUF_EXPECT_TOKENIZER_TOKENS),-expect-tokenizer-tokens $(GGUF_EXPECT_TOKENIZER_TOKENS),) $(if $(GGUF_EXPECT_BOS),-expect-bos $(GGUF_EXPECT_BOS),) $(if $(GGUF_EXPECT_EOS),-expect-eos $(GGUF_EXPECT_EOS),) $(if $(GGUF_EXPECT_MAX_SEQ_LEN),-expect-max-seq-len $(GGUF_EXPECT_MAX_SEQ_LEN),) $(if $(GGUF_EXPECT_FULL_ATTENTION_INTERVAL),-expect-full-attention-interval $(GGUF_EXPECT_FULL_ATTENTION_INTERVAL),) $(if $(GGUF_EXPECT_KV_HEADS),-expect-kv-heads $(GGUF_EXPECT_KV_HEADS),) $(if $(GGUF_EXPECT_HEAD_DIM),-expect-head-dim $(GGUF_EXPECT_HEAD_DIM),) $(if $(GGUF_EXPECT_KV_DIM),-expect-kv-dim $(GGUF_EXPECT_KV_DIM),) $(if $(GGUF_EXPECT_EXPERTS),-expect-experts $(GGUF_EXPECT_EXPERTS),) $(if $(GGUF_EXPECT_EXPERTS_PER_TOKEN),-expect-experts-per-token $(GGUF_EXPECT_EXPERTS_PER_TOKEN),) $(if $(GGUF_EXPECT_F32_COUNT),-expect-f32-count $(GGUF_EXPECT_F32_COUNT),) $(if $(GGUF_EXPECT_Q4_K_COUNT),-expect-q4-k-count $(GGUF_EXPECT_Q4_K_COUNT),) $(if $(GGUF_EXPECT_Q6_K_COUNT),-expect-q6-k-count $(GGUF_EXPECT_Q6_K_COUNT),) $(if $(GGUF_EXPECT_CACHE_LAYERS),-expect-cache-layers $(GGUF_EXPECT_CACHE_LAYERS),) $(if $(GGUF_EXPECT_PROTECTED_CACHE_LAYERS),-expect-protected-cache-layers $(GGUF_EXPECT_PROTECTED_CACHE_LAYERS),) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,) $(GGUF_MODEL)

gguf-smoke:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -prompt-ids $(GGUF_PROMPT_IDS) -max-new $(GGUF_MAX_NEW) -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_GENERATED),-expect-generated $(GGUF_EXPECT_GENERATED),) $(if $(GGUF_EXPECT_DECODED),-expect-decoded $(GGUF_EXPECT_DECODED),) $(if $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),-expect-runtime-float-bytes $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),-expect-runtime-compressed-bytes $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),-expect-runtime-scratch-bytes $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),-expect-runtime-total-bytes $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,)

gguf-bench:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -prompt-ids $(GGUF_PROMPT_IDS) -max-new $(GGUF_MAX_NEW) -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_GENERATED),-expect-generated $(GGUF_EXPECT_GENERATED),) $(if $(GGUF_EXPECT_DECODED),-expect-decoded $(GGUF_EXPECT_DECODED),) $(if $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),-expect-runtime-float-bytes $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),-expect-runtime-compressed-bytes $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),-expect-runtime-scratch-bytes $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),-expect-runtime-total-bytes $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),) $(if $(GGUF_EXPECT_KV_COMPRESSED_LAYERS),-expect-kv-compressed-layers $(GGUF_EXPECT_KV_COMPRESSED_LAYERS),) $(if $(GGUF_EXPECT_KV_SEQ),-expect-kv-seq $(GGUF_EXPECT_KV_SEQ),) $(if $(GGUF_EXPECT_KV_COMPRESSED_COUNT),-expect-kv-compressed-count $(GGUF_EXPECT_KV_COMPRESSED_COUNT),) $(if $(GGUF_EXPECT_KV_FULL_COUNT),-expect-kv-full-count $(GGUF_EXPECT_KV_FULL_COUNT),) $(if $(GGUF_EXPECT_KV_FLOAT_BYTES),-expect-kv-float-bytes $(GGUF_EXPECT_KV_FLOAT_BYTES),) $(if $(GGUF_EXPECT_KV_COMPRESSED_BYTES),-expect-kv-compressed-bytes $(GGUF_EXPECT_KV_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_KV_SCRATCH_BYTES),-expect-kv-scratch-bytes $(GGUF_EXPECT_KV_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_KV_TOTAL_BYTES),-expect-kv-total-bytes $(GGUF_EXPECT_KV_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,) -bench

gguf-turboquant-smoke:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -load-only -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) -kv-smoke-tokens $(GGUF_KV_SMOKE_TOKENS) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_LAYER),-expect-kv-smoke-layer $(GGUF_EXPECT_KV_SMOKE_LAYER),) $(if $(GGUF_EXPECT_KV_SMOKE_COMPRESSED),-expect-kv-smoke-compressed $(GGUF_EXPECT_KV_SMOKE_COMPRESSED),) $(if $(GGUF_EXPECT_KV_SMOKE_FULL),-expect-kv-smoke-full $(GGUF_EXPECT_KV_SMOKE_FULL),) $(if $(GGUF_EXPECT_KV_SMOKE_BYTES),-expect-kv-smoke-bytes $(GGUF_EXPECT_KV_SMOKE_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES),-expect-kv-smoke-scratch-bytes $(GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES),-expect-kv-smoke-total-bytes $(GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,)

gguf-validate: gguf-inspect gguf-smoke gguf-turboquant-smoke

gguf-check: gguf-validate gguf-bench

gguf-ci:
	go test $(GGUF_CI_PACKAGES) -run '^$$'
	$(MAKE) gguf-validate

gguf-inspect-qwen36-reap:
	$(MAKE) gguf-inspect GGUF_MODEL=$(GGUF_MODEL) GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_REAP_RATIO=0.20 GGUF_EXPECT_REAP_SOURCE=filename_or_name GGUF_EXPECT_ARCHITECTURE=qwen35moe GGUF_EXPECT_NAME_CONTAINS=REAP20 GGUF_EXPECT_TENSOR_COUNT=733 GGUF_EXPECT_LAYERS=40 GGUF_EXPECT_HIDDEN_SIZE=2048 GGUF_EXPECT_HEADS=16 GGUF_EXPECT_VOCAB_SIZE=248320 GGUF_EXPECT_TOKENIZER_TOKENS=248320 GGUF_EXPECT_BOS=248044 GGUF_EXPECT_EOS=248046 GGUF_EXPECT_MAX_SEQ_LEN=262144 GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 GGUF_EXPECT_KV_HEADS=2 GGUF_EXPECT_HEAD_DIM=256 GGUF_EXPECT_KV_DIM=512 GGUF_EXPECT_EXPERTS=205 GGUF_EXPECT_EXPERTS_PER_TOKEN=8 GGUF_EXPECT_F32_COUNT=301 GGUF_EXPECT_Q4_K_COUNT=371 GGUF_EXPECT_Q6_K_COUNT=61 GGUF_EXPECT_CACHE_LAYERS=10 GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 GGUF_EXPECT_FULL_KV_BYTES=10737418240 GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 GGUF_EXPECT_SAVED_KV_BYTES=8682143040 GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES=9663699456 GGUF_EXPECT_ESTIMATED_TOTAL_BYTES=11718974656 GGUF_EXPECT_SIMD_ROTATION=1

gguf-smoke-qwen36-reap:
	$(MAKE) gguf-smoke GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_SIMD_ROTATION=1

gguf-validate-qwen36-reap:
	$(MAKE) gguf-validate GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_REAP_RATIO=0.20 GGUF_EXPECT_REAP_SOURCE=filename_or_name GGUF_EXPECT_ARCHITECTURE=qwen35moe GGUF_EXPECT_NAME_CONTAINS=REAP20 GGUF_EXPECT_TENSOR_COUNT=733 GGUF_EXPECT_LAYERS=40 GGUF_EXPECT_HIDDEN_SIZE=2048 GGUF_EXPECT_HEADS=16 GGUF_EXPECT_VOCAB_SIZE=248320 GGUF_EXPECT_TOKENIZER_TOKENS=248320 GGUF_EXPECT_BOS=248044 GGUF_EXPECT_EOS=248046 GGUF_EXPECT_MAX_SEQ_LEN=262144 GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 GGUF_EXPECT_KV_HEADS=2 GGUF_EXPECT_HEAD_DIM=256 GGUF_EXPECT_KV_DIM=512 GGUF_EXPECT_EXPERTS=205 GGUF_EXPECT_EXPERTS_PER_TOKEN=8 GGUF_EXPECT_F32_COUNT=301 GGUF_EXPECT_Q4_K_COUNT=371 GGUF_EXPECT_Q6_K_COUNT=61 GGUF_EXPECT_CACHE_LAYERS=10 GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 GGUF_EXPECT_FULL_KV_BYTES=10737418240 GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 GGUF_EXPECT_SAVED_KV_BYTES=8682143040 GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES=9663699456 GGUF_EXPECT_ESTIMATED_TOTAL_BYTES=11718974656 GGUF_EXPECT_KV_SMOKE_LAYER=3 GGUF_EXPECT_KV_SMOKE_COMPRESSED=3 GGUF_EXPECT_KV_SMOKE_FULL=2 GGUF_EXPECT_KV_SMOKE_BYTES=9440 GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES=1280 GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES=10720 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_SIMD_ROTATION=1

gguf-bench-qwen36-reap:
	$(MAKE) gguf-bench GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_KV_COMPRESSED_LAYERS=10 GGUF_EXPECT_KV_SEQ=2 GGUF_EXPECT_KV_COMPRESSED_COUNT=0 GGUF_EXPECT_KV_FULL_COUNT=20 GGUF_EXPECT_KV_FLOAT_BYTES=245760 GGUF_EXPECT_KV_COMPRESSED_BYTES=81920 GGUF_EXPECT_KV_SCRATCH_BYTES=0 GGUF_EXPECT_KV_TOTAL_BYTES=327680 GGUF_EXPECT_SIMD_ROTATION=1

gguf-check-qwen36-reap: gguf-validate-qwen36-reap gguf-bench-qwen36-reap

gguf-ci-qwen36-reap:
	go test $(GGUF_CI_PACKAGES) -run '^$$'
	$(MAKE) gguf-check-qwen36-reap

QWEN3TTS_MODEL ?=
QWEN3TTS_TEXT ?= Hello world
QWEN3TTS_SPEAKER ?= ryan
QWEN3TTS_LANGUAGE ?= en
QWEN3TTS_INSPECT_FLAGS ?= -json
QWEN3TTS_FIXTURE ?= model/qwen3tts/testdata/customvoice_prompt_fixture.json
QWEN3TTS_FIXTURE_FLAGS ?= -json
LFM2_MODEL ?=
LFM2_INSPECT_FLAGS ?= -json
LFM2_FIXTURE ?= model/lfm2/testdata/lfm25_8b_a1b_metadata.json
LFM2_FIXTURE_FLAGS ?= -json

qwen3tts-inspect:
	@if [ -z "$(QWEN3TTS_MODEL)" ]; then echo "usage: make qwen3tts-inspect QWEN3TTS_MODEL=checkpoints/qwen3-tts-0.6b-customvoice [QWEN3TTS_TEXT='Hello world']"; exit 2; fi
	go run ./cmd/qwen/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -text "$(QWEN3TTS_TEXT)" -speaker $(QWEN3TTS_SPEAKER) -language $(QWEN3TTS_LANGUAGE) $(QWEN3TTS_INSPECT_FLAGS)

qwen3tts-fixture-coverage:
	@if [ -z "$(QWEN3TTS_MODEL)" ]; then echo "usage: make qwen3tts-fixture-coverage QWEN3TTS_MODEL=checkpoints/qwen3-tts-0.6b-customvoice [QWEN3TTS_FIXTURE=model/qwen3tts/testdata/customvoice_prompt_fixture.json]"; exit 2; fi
	go run ./cmd/qwen/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -fixture $(QWEN3TTS_FIXTURE) $(QWEN3TTS_FIXTURE_FLAGS)

lfm2-inspect:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-inspect LFM2_MODEL=checkpoints/lfm2.5-8b-a1b"; exit 2; fi
	go run ./cmd/models/lfm2inspect -model $(LFM2_MODEL) $(LFM2_INSPECT_FLAGS)

lfm2-fixture-coverage:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-fixture-coverage LFM2_MODEL=checkpoints/lfm2.5-8b-a1b [LFM2_FIXTURE=model/lfm2/testdata/lfm25_8b_a1b_metadata.json]"; exit 2; fi
	go run ./cmd/models/lfm2inspect -model $(LFM2_MODEL) -fixture $(LFM2_FIXTURE) $(LFM2_FIXTURE_FLAGS)

HUNYUAN3D_REPO ?= tencent/Hunyuan3D-2mini
HUNYUAN3D_SUBFOLDER ?= hunyuan3d-dit-v2-mini
HUNYUAN3D_INVENTORY ?= /workspace/tmp/hunyuan3d-mini-inventory.json
HUNYUAN3D_INVENTORY_FLAGS ?= --include-tensors
HUNYUAN3D_IMAGE_FIXTURE ?= /workspace/tmp/hunyuan3d-image-preprocess-fixture.json
HUNYUAN3D_IMAGE ?=
HUNYUAN3D_IMAGE_FLAGS ?=
HUNYUAN3D_SEAHORSE_IMAGE ?= testdata/hunyuan3d/seahorse_rgba.png
HUNYUAN3D_SEAHORSE_OUT ?= /workspace/tmp/hunyuan3d-seahorse.glb
HUNYUAN3D_SEAHORSE_FLAGS ?=
HUNYUAN3D_SRC ?= /workspace/tmp/Hunyuan3D-2-info
HUNYUAN3D_CONFIG ?=
HUNYUAN3D_CHECKPOINT ?=
HUNYUAN3D_INSPECT_FLAGS ?=
HUNYUAN3D_ENV_REPORT ?= /workspace/tmp/hunyuan3d-fixture-env.json
HUNYUAN3D_CONDITIONER_FIXTURE ?= /workspace/tmp/hunyuan3d-conditioner-fixture.json
HUNYUAN3D_CONDITIONER_FLAGS ?=
HUNYUAN3D_DENOISER_FIXTURE ?= /workspace/tmp/hunyuan3d-denoiser-step-fixture.json
HUNYUAN3D_DENOISER_FLAGS ?=
HUNYUAN3D_LOWSTEP_FIXTURE ?= /workspace/tmp/hunyuan3d-lowstep-latents-fixture.json
HUNYUAN3D_LOWSTEP_FLAGS ?=
HUNYUAN3D_MESH_FIXTURE ?= /workspace/tmp/hunyuan3d-mesh-fixture.json
HUNYUAN3D_MESH_FLAGS ?=
TRELLIS2_REPO ?= microsoft/TRELLIS.2-4B
TRELLIS2_REVISION ?= main
TRELLIS2_LOCAL_DIR ?=
TRELLIS2_INVENTORY ?= /workspace/tmp/trellis2-inventory.json
TRELLIS2_INVENTORY_FLAGS ?=
TRELLIS2_SRC ?= /workspace/tmp/TRELLIS.2
TRELLIS2_MODEL_DIR ?= microsoft/TRELLIS.2-4B
TRELLIS2_IMAGE ?=
TRELLIS2_ENV_REPORT ?= /workspace/tmp/trellis2-fixture-env.json
TRELLIS2_LOWSTEP_FIXTURE ?= /workspace/tmp/trellis2-lowstep-fixture.json
TRELLIS2_LOWSTEP_FLAGS ?=
TRELLIS2_OVOXEL_FILES ?=
TRELLIS2_OVOXEL_INSPECT ?= /workspace/tmp/trellis2-ovoxel-inspect.json

hunyuan3d-fixture-env:
	$(PYTHON) scripts/hunyuan3d_check_fixture_env.py --hunyuan3d-src $(HUNYUAN3D_SRC) $(if $(HUNYUAN3D_CONFIG),--config $(HUNYUAN3D_CONFIG),) $(if $(HUNYUAN3D_CHECKPOINT),--checkpoint $(HUNYUAN3D_CHECKPOINT),) $(if $(HUNYUAN3D_IMAGE),--image $(HUNYUAN3D_IMAGE),) --out $(HUNYUAN3D_ENV_REPORT)

hunyuan3d-inventory:
	$(PYTHON) scripts/hunyuan3d_fixture_inventory.py --repo $(HUNYUAN3D_REPO) --subfolder $(HUNYUAN3D_SUBFOLDER) --out $(HUNYUAN3D_INVENTORY) $(HUNYUAN3D_INVENTORY_FLAGS)

trellis2-fixture-env:
	$(PYTHON) scripts/trellis2_check_fixture_env.py --trellis2-src $(TRELLIS2_SRC) --model-dir $(TRELLIS2_MODEL_DIR) $(if $(TRELLIS2_IMAGE),--image $(TRELLIS2_IMAGE),) --out $(TRELLIS2_ENV_REPORT)

trellis2-inventory:
	$(PYTHON) scripts/trellis2_fixture_inventory.py --repo $(TRELLIS2_REPO) --revision $(TRELLIS2_REVISION) --out $(TRELLIS2_INVENTORY) $(if $(TRELLIS2_LOCAL_DIR),--local-dir $(TRELLIS2_LOCAL_DIR),) $(TRELLIS2_INVENTORY_FLAGS)

trellis2-lowstep-fixture:
	@if [ -z "$(TRELLIS2_IMAGE)" ]; then echo "usage: make trellis2-lowstep-fixture TRELLIS2_IMAGE=...png [TRELLIS2_SRC=/path/to/TRELLIS.2] [TRELLIS2_MODEL_DIR=/path/or/hf-id]"; exit 2; fi
	$(PYTHON) scripts/trellis2_lowstep_fixture.py --trellis2-src $(TRELLIS2_SRC) --model-dir $(TRELLIS2_MODEL_DIR) --image $(TRELLIS2_IMAGE) --out $(TRELLIS2_LOWSTEP_FIXTURE) $(TRELLIS2_LOWSTEP_FLAGS)

trellis2-ovoxel-inspect:
	@if [ -z "$(TRELLIS2_OVOXEL_FILES)" ]; then echo "usage: make trellis2-ovoxel-inspect TRELLIS2_OVOXEL_FILES='file1.npz file2.vxz'"; exit 2; fi
	$(PYTHON) scripts/trellis2_ovoxel_inspect.py --out $(TRELLIS2_OVOXEL_INSPECT) $(TRELLIS2_OVOXEL_FILES)

hunyuan3d-inspect:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ]; then echo "usage: make hunyuan3d-inspect HUNYUAN3D_CONFIG=.../config.yaml [HUNYUAN3D_CHECKPOINT=.../model.safetensors]"; exit 2; fi
	go run ./cmd/image/hy3dinspect -config $(HUNYUAN3D_CONFIG) $(if $(HUNYUAN3D_CHECKPOINT),-safetensors $(HUNYUAN3D_CHECKPOINT),) $(HUNYUAN3D_INSPECT_FLAGS)

hunyuan3d-seahorse:
	$(PYTHON) scripts/hunyuan3d_seahorse_demo.py --hunyuan3d-src $(HUNYUAN3D_SRC) --image $(HUNYUAN3D_SEAHORSE_IMAGE) --out $(HUNYUAN3D_SEAHORSE_OUT) --model $(HUNYUAN3D_REPO) --subfolder $(HUNYUAN3D_SUBFOLDER) $(HUNYUAN3D_SEAHORSE_FLAGS)

hunyuan3d-image-fixture:
	$(PYTHON) scripts/hunyuan3d_image_fixture.py --out $(HUNYUAN3D_IMAGE_FIXTURE) $(if $(HUNYUAN3D_IMAGE),--image $(HUNYUAN3D_IMAGE),) $(HUNYUAN3D_IMAGE_FLAGS)

hunyuan3d-conditioner-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-conditioner-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_conditioner_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_CONDITIONER_FIXTURE) $(HUNYUAN3D_CONDITIONER_FLAGS)

hunyuan3d-denoiser-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-denoiser-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_denoiser_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_DENOISER_FIXTURE) $(HUNYUAN3D_DENOISER_FLAGS)

hunyuan3d-lowstep-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-lowstep-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_lowstep_latent_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_LOWSTEP_FIXTURE) $(HUNYUAN3D_LOWSTEP_FLAGS)

hunyuan3d-mesh-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-mesh-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_mesh_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_MESH_FIXTURE) $(HUNYUAN3D_MESH_FLAGS)

# GPU-heavy tests (require GEMMA4_TRACE_TEST=1 and GPU)
test-gpu:
	GEMMA4_TRACE_TEST=1 go test -tags diagnostic -count=1 -run TestGemma4GPUBench ./model -v

# Quick smoke test
smoke:
	@echo "=== build ==="
	go build -o /dev/null ./cmd/llm/llmgen
	go build -o /dev/null ./cmd/llm/llmserver
	go build -o /dev/null ./cmd/llm/llmchat
	@echo "=== vet ==="
	go vet ./...
	@echo "=== unit tests ==="
	go test -count=1 -timeout=60s ./loader/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...
	@echo "=== ok ==="

# Ideogram 4 OSS/ComfyUI-style generation. ComfyUI-Ideogram4's recommended
# workflow is Magic Prompt -> Generate, where Magic Prompt produces a structured
# single-line JSON caption. Keep the prompt in a file and pass it verbatim so
# generation runs do not drift back to plain natural-language prompts.
IDEOGRAM4_MODEL ?= /srv/piclaw-dev/workspace/tmp/ideogram4-cat-model
IDEOGRAM4_PROMPT_FILE ?= prompts/ideogram4/cat.json
IDEOGRAM4_OUT ?= $(TMPDIR)/ideogram4/cat_comfy_prompt_256.png
IDEOGRAM4_WIDTH ?= 256
IDEOGRAM4_HEIGHT ?= 256
IDEOGRAM4_STEPS ?= 16
IDEOGRAM4_GUIDANCE ?= 7.0
IDEOGRAM4_MU ?= 0.0
IDEOGRAM4_STD ?= 1.75
IDEOGRAM4_SEED ?= 2026060803
IDEOGRAM4_GPU_RESIDENCY ?= phase
IDEOGRAM4_EXTRA_FLAGS ?= -gpu-fp8-sgemm
IDEOGRAM4_LAYER_CACHE_WINDOW ?= 21
IDEOGRAM4_LAYER_CACHE_START ?= 0
IDEOGRAM4_FULL_LAYER ?= 1
IDEOGRAM4_HIDDEN_RESIDENT ?= 1
IDEOGRAM4_LAYER_CACHE_ATTENTION_ALL ?= 0
IDEOGRAM4_LAYER_CACHE_WINDOW_COND ?= 34
IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND ?= 9
IDEOGRAM4_LAYER_CACHE_START_COND ?=
IDEOGRAM4_LAYER_CACHE_START_UNCOND ?=
IDEOGRAM4_LAYER_CACHE_QKV_ALL ?= 0
IDEOGRAM4_LAYER_CACHE_O_ALL ?= 0
IDEOGRAM4_ADALN_RESIDENT ?= 0
IDEOGRAM4_GPU_ENV ?= GO_PHERENCE_IDEOGRAM4_GPU_ADALN_RESIDENT=$(IDEOGRAM4_ADALN_RESIDENT) GO_PHERENCE_IDEOGRAM4_GPU_DIT_VECTOR=1 GO_PHERENCE_IDEOGRAM4_GPU_FULL_LAYER=$(IDEOGRAM4_FULL_LAYER) GO_PHERENCE_IDEOGRAM4_GPU_HIDDEN_RESIDENT=$(IDEOGRAM4_HIDDEN_RESIDENT) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW=$(IDEOGRAM4_LAYER_CACHE_WINDOW) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START=$(IDEOGRAM4_LAYER_CACHE_START) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_COND=$(IDEOGRAM4_LAYER_CACHE_WINDOW_COND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_UNCOND=$(IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START_COND=$(IDEOGRAM4_LAYER_CACHE_START_COND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START_UNCOND=$(IDEOGRAM4_LAYER_CACHE_START_UNCOND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_ATTENTION_ALL=$(IDEOGRAM4_LAYER_CACHE_ATTENTION_ALL) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_QKV_ALL=$(IDEOGRAM4_LAYER_CACHE_QKV_ALL) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_O_ALL=$(IDEOGRAM4_LAYER_CACHE_O_ALL)

.PHONY: ideogram4-cat-prompt ideogram4-cat-gpu ideogram4-cat-cpu ideogram4-cat-open

ideogram4-cat-prompt:
	$(PYTHON) -m json.tool $(IDEOGRAM4_PROMPT_FILE) >/dev/null
	@cat $(IDEOGRAM4_PROMPT_FILE)
	@echo

ideogram4-cat-gpu: ideogram4-cat-prompt
	mkdir -p $(dir $(IDEOGRAM4_OUT)) $(GOTMPDIR)
	$(IDEOGRAM4_GPU_ENV) go run ./cmd/image/ideogram4gen \
		-model $(IDEOGRAM4_MODEL) \
		-prompt "$$(cat $(IDEOGRAM4_PROMPT_FILE))" \
		-out $(IDEOGRAM4_OUT) \
		-width $(IDEOGRAM4_WIDTH) \
		-height $(IDEOGRAM4_HEIGHT) \
		-steps $(IDEOGRAM4_STEPS) \
		-guidance $(IDEOGRAM4_GUIDANCE) \
		-mu $(IDEOGRAM4_MU) \
		-std $(IDEOGRAM4_STD) \
		-seed $(IDEOGRAM4_SEED) \
		-gpu -gpu-fp8 -gpu-fp8-cache -gpu-residency $(IDEOGRAM4_GPU_RESIDENCY) \
		$(IDEOGRAM4_EXTRA_FLAGS) \
		-timing

ideogram4-cat-cpu: ideogram4-cat-prompt
	mkdir -p $(dir $(IDEOGRAM4_OUT)) $(GOTMPDIR)
	GO_PHERENCE_DISABLE_NVIDIA=1 go run ./cmd/image/ideogram4gen \
		-model $(IDEOGRAM4_MODEL) \
		-prompt "$$(cat $(IDEOGRAM4_PROMPT_FILE))" \
		-out $(IDEOGRAM4_OUT) \
		-width $(IDEOGRAM4_WIDTH) \
		-height $(IDEOGRAM4_HEIGHT) \
		-steps $(IDEOGRAM4_STEPS) \
		-guidance $(IDEOGRAM4_GUIDANCE) \
		-mu $(IDEOGRAM4_MU) \
		-std $(IDEOGRAM4_STD) \
		-seed $(IDEOGRAM4_SEED) \
		-timing

ideogram4-cat-open:
	@echo $(IDEOGRAM4_OUT)

IDEOGRAM4_SWEEP_STEPS ?= 2
IDEOGRAM4_SWEEP_WINDOWS ?= 0 2 4 8
IDEOGRAM4_SWEEP_STARTS ?= 0
IDEOGRAM4_SWEEP_CSV ?= $(TMPDIR)/ideogram4/residency_sweep.csv
.PHONY: ideogram4-residency-sweep
ideogram4-residency-sweep:
	IDEOGRAM4_SWEEP_STEPS='$(IDEOGRAM4_SWEEP_STEPS)' \
	IDEOGRAM4_SWEEP_WINDOWS='$(IDEOGRAM4_SWEEP_WINDOWS)' \
	IDEOGRAM4_SWEEP_STARTS='$(IDEOGRAM4_SWEEP_STARTS)' \
	IDEOGRAM4_SWEEP_CSV='$(IDEOGRAM4_SWEEP_CSV)' \
	IDEOGRAM4_MODEL='$(IDEOGRAM4_MODEL)' \
	IDEOGRAM4_PROMPT_FILE='$(IDEOGRAM4_PROMPT_FILE)' \
	IDEOGRAM4_WIDTH='$(IDEOGRAM4_WIDTH)' \
	IDEOGRAM4_HEIGHT='$(IDEOGRAM4_HEIGHT)' \
	IDEOGRAM4_GUIDANCE='$(IDEOGRAM4_GUIDANCE)' \
	IDEOGRAM4_MU='$(IDEOGRAM4_MU)' \
	IDEOGRAM4_STD='$(IDEOGRAM4_STD)' \
	IDEOGRAM4_SEED='$(IDEOGRAM4_SEED)' \
	IDEOGRAM4_GPU_RESIDENCY='$(IDEOGRAM4_GPU_RESIDENCY)' \
	IDEOGRAM4_FULL_LAYER='$(IDEOGRAM4_FULL_LAYER)' \
	./scripts/ideogram4_residency_sweep.sh

# Resolution-aware Ideogram presets for the local RTX 3060 12GB profile.
# 256px can use aggressive asymmetric residency. 512px needs reduced residency
# because the larger activation/attention buffers exceed VRAM with 256px defaults.
IDEOGRAM4_512_STEPS ?= 4
IDEOGRAM4_VAE_PROBE_HEIGHT ?= 512
IDEOGRAM4_VAE_PROBE_WIDTH ?= 512
.PHONY: ideogram4-cat-gpu-256 ideogram4-cat-gpu-512 ideogram4-vae-probe

ideogram4-cat-gpu-256:
	$(MAKE) ideogram4-cat-gpu \
		IDEOGRAM4_WIDTH=256 \
		IDEOGRAM4_HEIGHT=256 \
		IDEOGRAM4_LAYER_CACHE_WINDOW=21 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_COND=34 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND=9

ideogram4-cat-gpu-512:
	$(MAKE) ideogram4-cat-gpu \
		IDEOGRAM4_WIDTH=512 \
		IDEOGRAM4_HEIGHT=512 \
		IDEOGRAM4_STEPS=$(IDEOGRAM4_512_STEPS) \
		IDEOGRAM4_OUT=$(TMPDIR)/ideogram4/cat_comfy_prompt_512.png \
		IDEOGRAM4_LAYER_CACHE_WINDOW=0 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_COND=16 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND=0

ideogram4-vae-probe:
	go run ./cmd/image/ideogram4vaeprobe \
		-model $(IDEOGRAM4_MODEL) \
		-width $(IDEOGRAM4_VAE_PROBE_WIDTH) \
		-height $(IDEOGRAM4_VAE_PROBE_HEIGHT) \
		-gpu -gpu-stats

DIFFUSIONGEMMA_REPO ?= google/diffusiongemma-26B-A4B-it
DIFFUSIONGEMMA_MODEL ?= checkpoints/diffusiongemma-26B-A4B-it
DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD ?= no

.PHONY: diffusiongemma-download-metadata diffusiongemma-download

diffusiongemma-download-metadata:
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL) --metadata-only

diffusiongemma-download:
	@test "$(DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD)" = "yes" || (echo "Refusing ~48.10 GiB DiffusionGemma shard download. Re-run with DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes or use diffusiongemma-download-plan-only."; exit 2)
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL)

DIFFUSIONGEMMA_PROMPT_IDS ?= 2
DIFFUSIONGEMMA_PROMPT ?= hi
DIFFUSIONGEMMA_MAX_NEW ?= 16
DIFFUSIONGEMMA_CANVAS ?= 0
DIFFUSIONGEMMA_SEED ?= 1
DIFFUSIONGEMMA_ALLOW_SLOW_CPU ?= no
DIFFUSIONGEMMA_EAGER_MMAP ?= no
DIFFUSIONGEMMA_PRELOAD_GLOBALS ?= no
DIFFUSIONGEMMA_RESIDENT_LAYERS ?= 0
DIFFUSIONGEMMA_MOCK_TOKEN ?= 4
DIFFUSIONGEMMA_MOCK_TOKENS ?=
DIFFUSIONGEMMA_DENOISE_STEPS ?= 0
DIFFUSIONGEMMA_T_MIN ?= -1
DIFFUSIONGEMMA_T_MAX ?= -1
DIFFUSIONGEMMA_ENTROPY_BOUND ?= -1
DIFFUSIONGEMMA_STABILITY ?= -1
DIFFUSIONGEMMA_CONFIDENCE ?= -1

.PHONY: diffusiongemma-inspect diffusiongemma-inspect-json diffusiongemma-run-scaffold diffusiongemma-run-mock diffusiongemma-run-mock-json diffusiongemma-run-cpu diffusiongemma-run-cpu-json

diffusiongemma-inspect:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL)

diffusiongemma-inspect-json:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -json

diffusiongemma-run-scaffold:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt-ids $(DIFFUSIONGEMMA_PROMPT_IDS) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED)

diffusiongemma-run-mock:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) $(if $(DIFFUSIONGEMMA_MOCK_TOKENS),-mock-tokens $(DIFFUSIONGEMMA_MOCK_TOKENS),) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -decode

diffusiongemma-run-mock-json:
	mkdir -p $(dir $(DIFFUSIONGEMMA_RUN_OUT))
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) $(if $(DIFFUSIONGEMMA_MOCK_TOKENS),-mock-tokens $(DIFFUSIONGEMMA_MOCK_TOKENS),) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -decode -json > $(DIFFUSIONGEMMA_RUN_OUT)

diffusiongemma-run-cpu:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt-ids $(DIFFUSIONGEMMA_PROMPT_IDS) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher $(if $(filter yes,$(DIFFUSIONGEMMA_ALLOW_SLOW_CPU)),-allow-slow-cpu,) $(if $(filter yes,$(DIFFUSIONGEMMA_EAGER_MMAP)),-eager-mmap,) $(if $(filter yes,$(DIFFUSIONGEMMA_PRELOAD_GLOBALS)),-preload-globals,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RESIDENT_LAYERS)),-resident-layers $(DIFFUSIONGEMMA_RESIDENT_LAYERS),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB)),-residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS)),-max-dispatch-layers $(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS),) $(if $(filter yes,$(DIFFUSIONGEMMA_TAIL_AFTER_MAX_LAYERS)),-tail-after-max-layers,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_LM_HEAD_TOP_K)),-lm-head-top-k $(DIFFUSIONGEMMA_LM_HEAD_TOP_K),) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,)

diffusiongemma-run-cpu-json:
	mkdir -p $(dir $(DIFFUSIONGEMMA_RUN_OUT))
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt-ids $(DIFFUSIONGEMMA_PROMPT_IDS) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher $(if $(filter yes,$(DIFFUSIONGEMMA_ALLOW_SLOW_CPU)),-allow-slow-cpu,) $(if $(filter yes,$(DIFFUSIONGEMMA_EAGER_MMAP)),-eager-mmap,) $(if $(filter yes,$(DIFFUSIONGEMMA_PRELOAD_GLOBALS)),-preload-globals,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RESIDENT_LAYERS)),-resident-layers $(DIFFUSIONGEMMA_RESIDENT_LAYERS),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB)),-residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS)),-max-dispatch-layers $(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS),) $(if $(filter yes,$(DIFFUSIONGEMMA_TAIL_AFTER_MAX_LAYERS)),-tail-after-max-layers,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_LM_HEAD_TOP_K)),-lm-head-top-k $(DIFFUSIONGEMMA_LM_HEAD_TOP_K),) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -json > $(DIFFUSIONGEMMA_RUN_OUT)

DIFFUSIONGEMMA_STATUS_OUT ?= $(TMPDIR)/diffusiongemma/status.json
DIFFUSIONGEMMA_REF_OUT ?= $(TMPDIR)/diffusiongemma/reference.json
DIFFUSIONGEMMA_REF_PROMPT ?= Why is the sky blue?
DIFFUSIONGEMMA_REF_MESSAGES_JSON ?=
DIFFUSIONGEMMA_REF_MESSAGES_FILE ?=
DIFFUSIONGEMMA_REF_MAX_NEW ?= 64
DIFFUSIONGEMMA_REF_STEPS ?= 48

.PHONY: diffusiongemma-reference-dry-run diffusiongemma-reference

diffusiongemma-reference-dry-run:
	python3 scripts/diffusiongemma_reference.py --model $(DIFFUSIONGEMMA_MODEL) --dry-run --out $(DIFFUSIONGEMMA_REF_OUT)

diffusiongemma-reference:
	python3 scripts/diffusiongemma_reference.py --model $(DIFFUSIONGEMMA_MODEL) --prompt '$(DIFFUSIONGEMMA_REF_PROMPT)' $(if $(DIFFUSIONGEMMA_REF_MESSAGES_JSON),--messages-json '$(DIFFUSIONGEMMA_REF_MESSAGES_JSON)',) $(if $(DIFFUSIONGEMMA_REF_MESSAGES_FILE),--messages-file $(DIFFUSIONGEMMA_REF_MESSAGES_FILE),) --max-new-tokens $(DIFFUSIONGEMMA_REF_MAX_NEW) --max-denoising-steps $(DIFFUSIONGEMMA_REF_STEPS) --out $(DIFFUSIONGEMMA_REF_OUT)

.PHONY: diffusiongemma-check-scaffold diffusiongemma-golden-gate

diffusiongemma-check-scaffold:
	mkdir -p $(GOTMPDIR)
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -require-text-scaffold-ready
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) $(if $(DIFFUSIONGEMMA_MOCK_TOKENS),-mock-tokens $(DIFFUSIONGEMMA_MOCK_TOKENS),) -canvas 2 -max-new 2 -decode
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -messages-json '[{"role":"user","content":"$(DIFFUSIONGEMMA_PROMPT)"}]' -add-bos -chat-template -generation-prompt -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) $(if $(DIFFUSIONGEMMA_MOCK_TOKENS),-mock-tokens $(DIFFUSIONGEMMA_MOCK_TOKENS),) -canvas 2 -max-new 2 -decode
	go test ./cmd/diffusiongemmarun ./cmd/diffusiongemmainspect ./model/diffusiongemma ./loader/config -run '^$$'

diffusiongemma-golden-gate:
	go test ./model/diffusiongemma -run 'TestLlamaCppGGUFHi1x1(GoldenResponseIDs|ReferenceFixture)|TestGGUFHi1x1(GoTrimmedOutputComparisonGate|TopLogitProbeGate|ParityStatusDocumentsCurrentBlocker)|TestGGUFHiPhaseAlignedTrace|TestGGUFHiPhaseAlignedInputNormParityGate|TestGGUFHiPhaseAlignedLayer0OpsParityGate|TestMT19937|TestGGUFQ(4K|8_0|5_0)ExpertRowDotMatchesScalarDequantOracle|TestGGUFQ6KLMHeadRowDotMatchesQ8KRoundedDequantOracle|TestQ6I8DotAVX2MatchesScalar' -count=1 -v
	DIFFUSIONGEMMA_LOCAL_GGUF_GOLDEN_EXIT=1 go test ./model/diffusiongemma -run 'TestLocalGGUFTinyForwardGoldenTopLogits' -count=1 -v
	flock /tmp/go-pherence-gpu.lock -c "go test ./model/diffusiongemma -run 'TestLocalGGUFGPUCPUCanvas1Parity' -count=1 -v"

.PHONY: diffusiongemma-ci-scaffold

diffusiongemma-ci-scaffold: diffusiongemma-check-scaffold diffusiongemma-golden-gate diffusiongemma-reference-dry-run diffusiongemma-mock-compare diffusiongemma-ci-structured-messages diffusiongemma-status-json diffusiongemma-status-summary

.PHONY: diffusiongemma-check-shards

diffusiongemma-check-shards:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -require-shards-ready

DIFFUSIONGEMMA_RUN_OUT ?= $(TMPDIR)/diffusiongemma/run.json
DIFFUSIONGEMMA_MOCK_REF_OUT ?= $(TMPDIR)/diffusiongemma/mock_reference.json
DIFFUSIONGEMMA_COMPARE_PREFIX ?= 0

.PHONY: diffusiongemma-compare-reference

diffusiongemma-compare-reference:
	python3 scripts/diffusiongemma_compare_reference.py --reference $(DIFFUSIONGEMMA_REF_OUT) --run $(DIFFUSIONGEMMA_RUN_OUT) --prefix $(DIFFUSIONGEMMA_COMPARE_PREFIX)

.PHONY: diffusiongemma-mock-compare

diffusiongemma-mock-compare:
	mkdir -p $(dir $(DIFFUSIONGEMMA_MOCK_REF_OUT)) $(dir $(DIFFUSIONGEMMA_RUN_OUT))
	python3 -c 'import json, sys; out,toks,token=sys.argv[1:4]; toks=toks or (token+","+token); ids=[int(x.strip()) for x in toks.split(",") if x.strip()]; json.dump({"output_ids": ids[:2]}, open(out, "w"))' $(DIFFUSIONGEMMA_MOCK_REF_OUT) '$(DIFFUSIONGEMMA_MOCK_TOKENS)' $(DIFFUSIONGEMMA_MOCK_TOKEN)
	$(MAKE) diffusiongemma-run-mock-json DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_PROMPT='$(DIFFUSIONGEMMA_PROMPT)' DIFFUSIONGEMMA_MOCK_TOKEN=$(DIFFUSIONGEMMA_MOCK_TOKEN) DIFFUSIONGEMMA_MOCK_TOKENS=$(DIFFUSIONGEMMA_MOCK_TOKENS) DIFFUSIONGEMMA_CANVAS=2 DIFFUSIONGEMMA_MAX_NEW=2 DIFFUSIONGEMMA_RUN_OUT=$(DIFFUSIONGEMMA_RUN_OUT)
	$(MAKE) diffusiongemma-compare-reference DIFFUSIONGEMMA_REF_OUT=$(DIFFUSIONGEMMA_MOCK_REF_OUT) DIFFUSIONGEMMA_RUN_OUT=$(DIFFUSIONGEMMA_RUN_OUT)

.PHONY: diffusiongemma-status-json

diffusiongemma-status-json:
	mkdir -p $(dir $(DIFFUSIONGEMMA_STATUS_OUT))
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -json > $(DIFFUSIONGEMMA_STATUS_OUT)

.PHONY: diffusiongemma-status-summary

diffusiongemma-status-summary:
	python3 scripts/diffusiongemma_status_summary.py $(DIFFUSIONGEMMA_STATUS_OUT)

.PHONY: diffusiongemma-ci-mock-pattern

diffusiongemma-ci-mock-pattern:
	$(MAKE) diffusiongemma-mock-compare DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_PROMPT='$(DIFFUSIONGEMMA_PROMPT)' DIFFUSIONGEMMA_MOCK_TOKENS=4,2 DIFFUSIONGEMMA_RUN_OUT=$(TMPDIR)/diffusiongemma/mock_pattern_run.json DIFFUSIONGEMMA_MOCK_REF_OUT=$(TMPDIR)/diffusiongemma/mock_pattern_ref.json

.PHONY: diffusiongemma-check-weights

diffusiongemma-check-weights:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -require-shards-ready -open-weights

.PHONY: diffusiongemma-parity

diffusiongemma-parity: diffusiongemma-weights-json diffusiongemma-reference diffusiongemma-run-cpu-json diffusiongemma-compare-reference

.PHONY: diffusiongemma-bootstrap-scaffold

diffusiongemma-bootstrap-scaffold: diffusiongemma-download-metadata diffusiongemma-ci-scaffold

DIFFUSIONGEMMA_MESSAGES_JSON ?= [{"role":"user","content":"$(DIFFUSIONGEMMA_PROMPT)"}]

.PHONY: diffusiongemma-ci-structured-messages

diffusiongemma-ci-structured-messages:
	$(MAKE) diffusiongemma-reference-dry-run DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_REF_OUT=$(DIFFUSIONGEMMA_REF_OUT) DIFFUSIONGEMMA_REF_MESSAGES_JSON='$(DIFFUSIONGEMMA_MESSAGES_JSON)'
	$(MAKE) diffusiongemma-run-mock-json DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_MOCK_TOKEN=$(DIFFUSIONGEMMA_MOCK_TOKEN) DIFFUSIONGEMMA_MOCK_TOKENS=$(DIFFUSIONGEMMA_MOCK_TOKENS) DIFFUSIONGEMMA_CANVAS=2 DIFFUSIONGEMMA_MAX_NEW=2 DIFFUSIONGEMMA_RUN_OUT=$(DIFFUSIONGEMMA_RUN_OUT) DIFFUSIONGEMMA_PROMPT='$(DIFFUSIONGEMMA_PROMPT)'
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -messages-json '$(DIFFUSIONGEMMA_MESSAGES_JSON)' -add-bos -chat-template -generation-prompt -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) $(if $(DIFFUSIONGEMMA_MOCK_TOKENS),-mock-tokens $(DIFFUSIONGEMMA_MOCK_TOKENS),) -canvas 2 -max-new 2 -decode

.PHONY: diffusiongemma-download-plan

diffusiongemma-download-plan: diffusiongemma-download-metadata diffusiongemma-inspect diffusiongemma-status-json diffusiongemma-status-summary

.PHONY: diffusiongemma-status-refresh

diffusiongemma-status-refresh: diffusiongemma-download-metadata diffusiongemma-status-json diffusiongemma-status-summary

.PHONY: diffusiongemma-download-plan-only

diffusiongemma-download-plan-only:
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL) --plan-only

DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT ?= $(TMPDIR)/diffusiongemma/download_plan.json

.PHONY: diffusiongemma-download-plan-json

diffusiongemma-download-plan-json:
	mkdir -p $(dir $(DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT))
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL) --plan-only --json-plan > $(DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT)

.PHONY: diffusiongemma-download-plan-summary

diffusiongemma-download-plan-summary:
	python3 scripts/diffusiongemma_download_plan_summary.py $(DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT)

.PHONY: diffusiongemma-download-plan-report

diffusiongemma-download-plan-report: diffusiongemma-download-plan-json diffusiongemma-download-plan-summary

.PHONY: diffusiongemma-download-status

diffusiongemma-download-status: diffusiongemma-status-refresh

DIFFUSIONGEMMA_WEIGHTS_OUT ?= $(TMPDIR)/diffusiongemma/weights.json

.PHONY: diffusiongemma-weights-json

diffusiongemma-weights-json: diffusiongemma-check-shards
	mkdir -p $(dir $(DIFFUSIONGEMMA_WEIGHTS_OUT))
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -open-weights -json > $(DIFFUSIONGEMMA_WEIGHTS_OUT)

.PHONY: diffusiongemma-ci-no-weights

diffusiongemma-ci-no-weights: diffusiongemma-download-plan-report diffusiongemma-reference-env diffusiongemma-ci-scaffold diffusiongemma-ci-mock-pattern

.PHONY: diffusiongemma-help

diffusiongemma-help:
	@echo "DiffusionGemma safe/no-weight workflow:"
	@echo "  make diffusiongemma-download-plan-report"
	@echo "  make diffusiongemma-status-refresh"
	@echo "  make diffusiongemma-ci-no-weights"
	@echo "  make diffusiongemma-ci-sparse-text DIFFUSIONGEMMA_MODEL=checkpoints/diffusiongemma-26B-A4B-it"
	@echo ""
	@echo "DiffusionGemma full checkpoint workflow (~48.10 GiB):"
	@echo "  make diffusiongemma-download DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes"
	@echo "  make diffusiongemma-check-shards"
	@echo "  make diffusiongemma-check-weights"
	@echo ""
	@echo "DiffusionGemma parity workflow (requires full shards + Transformers/PyTorch):"
	@echo "  make diffusiongemma-parity"

DIFFUSIONGEMMA_ENV_OUT ?= $(TMPDIR)/diffusiongemma/reference_env.json

.PHONY: diffusiongemma-reference-env

diffusiongemma-reference-env:
	mkdir -p $(dir $(DIFFUSIONGEMMA_ENV_OUT))
	python3 scripts/diffusiongemma_reference.py --check-env --out $(DIFFUSIONGEMMA_ENV_OUT)

DIFFUSIONGEMMA_CPU_SMOKE_PROMPT ?= hi
DIFFUSIONGEMMA_CPU_SMOKE_CANVAS ?= 1
DIFFUSIONGEMMA_CPU_SMOKE_MAX_NEW ?= 1

.PHONY: diffusiongemma-run-cpu-smoke

diffusiongemma-run-cpu-smoke: diffusiongemma-check-weights
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_CPU_SMOKE_PROMPT)' -max-new $(DIFFUSIONGEMMA_CPU_SMOKE_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CPU_SMOKE_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher $(if $(filter yes,$(DIFFUSIONGEMMA_ALLOW_SLOW_CPU)),-allow-slow-cpu,) $(if $(filter yes,$(DIFFUSIONGEMMA_EAGER_MMAP)),-eager-mmap,) $(if $(filter yes,$(DIFFUSIONGEMMA_PRELOAD_GLOBALS)),-preload-globals,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RESIDENT_LAYERS)),-resident-layers $(DIFFUSIONGEMMA_RESIDENT_LAYERS),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB)),-residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB),) $(if $(filter-out 0,$(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS)),-max-dispatch-layers $(DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS),) $(if $(filter yes,$(DIFFUSIONGEMMA_TAIL_AFTER_MAX_LAYERS)),-tail-after-max-layers,) $(if $(filter-out 0,$(DIFFUSIONGEMMA_LM_HEAD_TOP_K)),-lm-head-top-k $(DIFFUSIONGEMMA_LM_HEAD_TOP_K),) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -decode

IDEOGRAM4_K3_HANDOFF_DIR ?= $(TMPDIR)/ideogram4/k3-handoff
.PHONY: ideogram4-k3-handoff
ideogram4-k3-handoff: ideogram4-k3-check
	IDEOGRAM4_K3_HANDOFF_DIR='$(IDEOGRAM4_K3_HANDOFF_DIR)' ./scripts/ideogram4_k3_handoff.sh

.PHONY: diffusiongemma-preload-globals

diffusiongemma-preload-globals: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -cpu-dispatcher -allow-slow-cpu -preload-globals -preload-only

.PHONY: diffusiongemma-preload-layer0

diffusiongemma-preload-layer0: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -cpu-dispatcher -allow-slow-cpu -preload-globals -resident-layers 1 -preload-only

DIFFUSIONGEMMA_RESIDENCY_OUT ?= $(TMPDIR)/diffusiongemma/residency.json
DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB ?= 0
DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB ?= 16
DIFFUSIONGEMMA_MAX_DISPATCH_LAYERS ?= 0
DIFFUSIONGEMMA_TAIL_AFTER_MAX_LAYERS ?= no
DIFFUSIONGEMMA_LM_HEAD_TOP_K ?= 0
DIFFUSIONGEMMA_SPARSE_LM_HEAD_TOP_K ?= 8
DIFFUSIONGEMMA_EXPECT_GENERATED ?= 147485
DIFFUSIONGEMMA_DISPATCH_PROGRESS ?= no

.PHONY: diffusiongemma-residency-plan

diffusiongemma-residency-plan: diffusiongemma-check-shards
	mkdir -p $(dir $(DIFFUSIONGEMMA_RESIDENCY_OUT))
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -open-weights -resident-layers $(DIFFUSIONGEMMA_RESIDENT_LAYERS) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB)),-residency-budget-gib $(DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB),) -json > $(DIFFUSIONGEMMA_RESIDENCY_OUT)
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -open-weights -resident-layers $(DIFFUSIONGEMMA_RESIDENT_LAYERS) $(if $(filter-out 0,$(DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB)),-residency-budget-gib $(DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB),) | grep residency

.PHONY: diffusiongemma-run-cpu-layer1-smoke

diffusiongemma-run-cpu-layer1-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -resident-layers 1 -max-dispatch-layers 1 -decode

.PHONY: diffusiongemma-run-cpu-layer2-evict-smoke

diffusiongemma-run-cpu-layer2-evict-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -resident-layers 1 -max-dispatch-layers 2 -decode

.PHONY: diffusiongemma-run-cpu-layer4-budget-smoke

diffusiongemma-run-cpu-layer4-budget-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 4 -decode

.PHONY: diffusiongemma-run-cpu-layer8-budget-smoke

diffusiongemma-run-cpu-layer8-budget-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 8 -decode

.PHONY: diffusiongemma-run-cpu-layer16-budget-smoke

diffusiongemma-run-cpu-layer16-budget-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 16 -decode

.PHONY: diffusiongemma-run-cpu-layer1-topk-smoke

diffusiongemma-run-cpu-layer1-topk-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -resident-layers 1 -max-dispatch-layers 1 -tail-after-max-layers -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-layer4-topk-smoke

diffusiongemma-run-cpu-layer4-topk-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 4 -tail-after-max-layers -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-layer8-topk-smoke

diffusiongemma-run-cpu-layer8-topk-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 8 -tail-after-max-layers -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-layer16-topk-step-smoke

diffusiongemma-run-cpu-layer16-topk-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 16 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-layer30-topk-step-smoke

diffusiongemma-run-cpu-layer30-topk-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 30 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-layer30-topk-2step-smoke

diffusiongemma-run-cpu-layer30-topk-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -max-dispatch-layers 30 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-step-smoke

diffusiongemma-run-cpu-full-topk-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-2step-smoke

diffusiongemma-run-cpu-full-topk-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-4step-smoke

diffusiongemma-run-cpu-full-topk-4step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 4 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-8step-smoke

diffusiongemma-run-cpu-full-topk-8step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 1 -canvas 1 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 8 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas2-step-smoke

diffusiongemma-run-cpu-full-topk-canvas2-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 2 -canvas 2 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas2-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas2-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 2 -canvas 2 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas4-step-smoke

diffusiongemma-run-cpu-full-topk-canvas4-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 4 -canvas 4 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas4-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas4-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 4 -canvas 4 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas8-step-smoke

diffusiongemma-run-cpu-full-topk-canvas8-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 8 -canvas 8 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas8-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas8-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 8 -canvas 8 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas16-step-smoke

diffusiongemma-run-cpu-full-topk-canvas16-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 16 -canvas 16 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas16-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas16-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 16 -canvas 16 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas32-step-smoke

diffusiongemma-run-cpu-full-topk-canvas32-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 32 -canvas 32 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas32-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas32-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 32 -canvas 32 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas64-step-smoke

diffusiongemma-run-cpu-full-topk-canvas64-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 64 -canvas 64 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas64-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas64-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 64 -canvas 64 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas128-step-smoke

diffusiongemma-run-cpu-full-topk-canvas128-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 128 -canvas 128 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas128-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas128-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 128 -canvas 128 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas256-step-smoke

diffusiongemma-run-cpu-full-topk-canvas256-step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 256 -canvas 256 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-run-cpu-full-topk-canvas256-2step-smoke

diffusiongemma-run-cpu-full-topk-canvas256-2step-smoke: diffusiongemma-check-shards
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new 256 -canvas 256 -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps 2 -cpu-dispatcher -allow-slow-cpu -residency-budget-gib 16 -lm-head-top-k 8 -decode

.PHONY: diffusiongemma-check-sparse-text

diffusiongemma-check-sparse-text:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -require-text-sparse-ready

.PHONY: diffusiongemma-ci-sparse-text

diffusiongemma-ci-sparse-text: diffusiongemma-check-sparse-text diffusiongemma-residency-plan diffusiongemma-run-sparse-text-json-check diffusiongemma-run-cpu-full-topk-step-smoke diffusiongemma-run-cpu-full-topk-canvas8-2step-smoke diffusiongemma-run-sparse-chat-json

.PHONY: diffusiongemma-ci-sparse-text-published

diffusiongemma-ci-sparse-text-published: diffusiongemma-check-sparse-text diffusiongemma-residency-plan diffusiongemma-run-cpu-full-topk-canvas256-step-smoke diffusiongemma-run-cpu-full-topk-canvas256-2step-smoke

.PHONY: diffusiongemma-run-sparse-text

diffusiongemma-run-sparse-text: diffusiongemma-check-sparse-text
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB) -lm-head-top-k $(DIFFUSIONGEMMA_SPARSE_LM_HEAD_TOP_K) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -decode

.PHONY: diffusiongemma-run-sparse-text-json

diffusiongemma-run-sparse-text-json: diffusiongemma-check-sparse-text
	mkdir -p $(dir $(DIFFUSIONGEMMA_RUN_OUT))
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB) -lm-head-top-k $(DIFFUSIONGEMMA_SPARSE_LM_HEAD_TOP_K) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -decode -json > $(DIFFUSIONGEMMA_RUN_OUT)
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("result") or {}; print("generated="+str(r.get("generated"))); print("error="+str(d.get("error")))' $(DIFFUSIONGEMMA_RUN_OUT)

.PHONY: diffusiongemma-run-sparse-chat-text

diffusiongemma-run-sparse-chat-text: diffusiongemma-check-sparse-text
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -messages-json '$(DIFFUSIONGEMMA_MESSAGES_JSON)' -add-bos -chat-template -generation-prompt -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB) -lm-head-top-k $(DIFFUSIONGEMMA_SPARSE_LM_HEAD_TOP_K) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -decode

.PHONY: diffusiongemma-run-sparse-chat-json

diffusiongemma-run-sparse-chat-json: diffusiongemma-check-sparse-text
	mkdir -p $(dir $(DIFFUSIONGEMMA_RUN_OUT))
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -messages-json '$(DIFFUSIONGEMMA_MESSAGES_JSON)' -add-bos -chat-template -generation-prompt -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -denoise-steps $(DIFFUSIONGEMMA_DENOISE_STEPS) -t-min $(DIFFUSIONGEMMA_T_MIN) -t-max $(DIFFUSIONGEMMA_T_MAX) -entropy-bound $(DIFFUSIONGEMMA_ENTROPY_BOUND) -stability $(DIFFUSIONGEMMA_STABILITY) -confidence $(DIFFUSIONGEMMA_CONFIDENCE) -cpu-dispatcher -allow-slow-cpu -residency-budget-gib $(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB) -lm-head-top-k $(DIFFUSIONGEMMA_SPARSE_LM_HEAD_TOP_K) $(if $(filter yes,$(DIFFUSIONGEMMA_DISPATCH_PROGRESS)),-dispatch-progress,) -decode -json > $(DIFFUSIONGEMMA_RUN_OUT)
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("result") or {}; print("generated="+str(r.get("generated"))); print("error="+str(d.get("error")))' $(DIFFUSIONGEMMA_RUN_OUT)

.PHONY: diffusiongemma-compare-sparse-run

diffusiongemma-compare-sparse-run:
	python3 scripts/diffusiongemma_compare_sparse_run.py $(DIFFUSIONGEMMA_RUN_OUT) --expected '$(DIFFUSIONGEMMA_EXPECT_GENERATED)'

.PHONY: diffusiongemma-run-sparse-text-json-check

diffusiongemma-run-sparse-text-json-check:
	$(MAKE) diffusiongemma-run-sparse-text-json DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_PROMPT='$(DIFFUSIONGEMMA_PROMPT)' DIFFUSIONGEMMA_MAX_NEW=$(DIFFUSIONGEMMA_MAX_NEW) DIFFUSIONGEMMA_CANVAS=$(DIFFUSIONGEMMA_CANVAS) DIFFUSIONGEMMA_DENOISE_STEPS=$(DIFFUSIONGEMMA_DENOISE_STEPS) DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB=$(DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB) DIFFUSIONGEMMA_RUN_OUT=$(DIFFUSIONGEMMA_RUN_OUT)
	$(MAKE) diffusiongemma-compare-sparse-run DIFFUSIONGEMMA_RUN_OUT=$(DIFFUSIONGEMMA_RUN_OUT) DIFFUSIONGEMMA_EXPECT_GENERATED='$(DIFFUSIONGEMMA_EXPECT_GENERATED)'

.PHONY: diffusiongemma-ci-sparse-text-fast

diffusiongemma-ci-sparse-text-fast: diffusiongemma-check-sparse-text diffusiongemma-residency-plan
	$(MAKE) diffusiongemma-run-sparse-text-json-check DIFFUSIONGEMMA_MODEL=$(DIFFUSIONGEMMA_MODEL) DIFFUSIONGEMMA_PROMPT=hi DIFFUSIONGEMMA_MAX_NEW=1 DIFFUSIONGEMMA_CANVAS=1 DIFFUSIONGEMMA_DENOISE_STEPS=1 DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB=16 DIFFUSIONGEMMA_RUN_OUT=$(TMPDIR)/diffusiongemma/ci_sparse_fast.json DIFFUSIONGEMMA_EXPECT_GENERATED=147485
	go test ./cmd/diffusiongemmarun ./cmd/diffusiongemmainspect ./model/diffusiongemma ./loader/config -run '^$$'

.PHONY: diffusiongemma-k3-profile diffusiongemma-k3-smoke diffusiongemma-k3-fp8-fallback-smoke diffusiongemma-k3-check
DIFFUSIONGEMMA_K3_MODEL ?= /home/me/models/diffusiongemma-26B-A4B-it-FP8
DIFFUSIONGEMMA_K3_CANVAS ?= 16
DIFFUSIONGEMMA_K3_STEPS ?= 2
DIFFUSIONGEMMA_K3_Q80_BUDGET_GIB ?= 2.0
DIFFUSIONGEMMA_K3_RETAIN_SELECTED_EXPERT_LAYERS ?= 30
DIFFUSIONGEMMA_K3_MAX_DISPATCH_LAYERS ?= 0
DIFFUSIONGEMMA_K3_TAIL_AFTER_MAX_LAYERS ?= 0
DIFFUSIONGEMMA_K3_SKIP_EVICTION ?= 0
DIFFUSIONGEMMA_K3_SMOKE_EXPECT_GENERATED ?= 239683

diffusiongemma-k3-profile: TMPDIR := /tmp
diffusiongemma-k3-profile: GOTMPDIR := /tmp
diffusiongemma-k3-profile:
	MODEL=$(DIFFUSIONGEMMA_K3_MODEL) CANVAS=$(DIFFUSIONGEMMA_K3_CANVAS) STEPS=$(DIFFUSIONGEMMA_K3_STEPS) Q80_BUDGET_GIB=$(DIFFUSIONGEMMA_K3_Q80_BUDGET_GIB) RETAIN_SELECTED_EXPERT_LAYERS=$(DIFFUSIONGEMMA_K3_RETAIN_SELECTED_EXPERT_LAYERS) MAX_DISPATCH_LAYERS=$(DIFFUSIONGEMMA_K3_MAX_DISPATCH_LAYERS) TAIL_AFTER_MAX_LAYERS=$(DIFFUSIONGEMMA_K3_TAIL_AFTER_MAX_LAYERS) SKIP_EVICTION=$(DIFFUSIONGEMMA_K3_SKIP_EVICTION) ./scripts/diffusiongemma_k3_profile.sh

diffusiongemma-k3-smoke: TMPDIR := /tmp
diffusiongemma-k3-smoke: GOTMPDIR := /tmp
diffusiongemma-k3-smoke:
	MODEL=$(DIFFUSIONGEMMA_K3_MODEL) CANVAS=1 STEPS=1 MAX_NEW=1 Q80_BUDGET_GIB=0.2 RETAIN_SELECTED_EXPERT_LAYERS=1 MAX_DISPATCH_LAYERS=1 TAIL_AFTER_MAX_LAYERS=1 EXPECT_GENERATED=$(DIFFUSIONGEMMA_K3_SMOKE_EXPECT_GENERATED) TAG=k3-smoke ./scripts/diffusiongemma_k3_profile.sh

diffusiongemma-k3-fp8-fallback-smoke: TMPDIR := /tmp
diffusiongemma-k3-fp8-fallback-smoke: GOTMPDIR := /tmp
diffusiongemma-k3-fp8-fallback-smoke:
	GO_PHERENCE_DIFFUSIONGEMMA_K3=1 GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8=0 HOME=/home/me TMPDIR=/tmp GOCACHE=/home/me/.cache/go-build GOMODCACHE=/home/me/go/pkg/mod go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_K3_MODEL) -prompt-ids 2,3 -max-new 1 -canvas 1 -denoise-steps 1 -cpu-dispatcher -allow-slow-cpu -max-dispatch-layers 1 -lm-head-top-k 8 -json > /tmp/diffusiongemma-k3-fp8-fallback-smoke.json
	jq -e '.result.generated == [0]' /tmp/diffusiongemma-k3-fp8-fallback-smoke.json >/dev/null

diffusiongemma-k3-check: diffusiongemma-k3-smoke diffusiongemma-k3-fp8-fallback-smoke
	HOME=/home/me TMPDIR=/tmp GOTMPDIR=/tmp GOCACHE=/home/me/.cache/go-build GOMODCACHE=/home/me/go/pkg/mod go test ./model/diffusiongemma ./cmd/diffusiongemmarun ./backends/spacemit/aicpu/aipool ./backends/spacemit/ime2
	HOME=/home/me TMPDIR=/tmp GOTMPDIR=/tmp GOCACHE=/home/me/.cache/go-build GOMODCACHE=/home/me/go/pkg/mod GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL=$(DIFFUSIONGEMMA_K3_MODEL) go test -run 'TestCachedFloatTensorAppliesFP8Scale|TestK3A100Q80ModelProjection' -v ./model/diffusiongemma
	@echo "DiffusionGemma K3 check passed: A100 smoke + scaled-FP8 fallback smoke + model-backed Q80/FP8 tests + affected Go tests"
