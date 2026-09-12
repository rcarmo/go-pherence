# Experimental trained Community-1 segmentation

The Go-owned PCM → SincNet → LSTM → powerset-head path now runs the pinned Community-1 segmentation checkpoint. All four final-output and local speaker-mask comparisons pass on the tested inputs. Two intermediate comparisons fail, so the path remains explicitly experimental and is not selected by any default.

## Checkpoint and conversion

- Model: `pyannote/speaker-diarization-community-1`, revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.
- Raw `segmentation/pytorch_model.bin`: SHA256 `7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e`, 5,906,507 bytes.
- Architecture: mono16k, SincNet stride10, four monolithic bidirectional IFGO LSTM layers with 128 hidden units, two 128-unit Linear/leaky-ReLU layers, seven log-softmax powerset classes (three local speakers, at most two active).
- Exact raw inventory: 54 F32 tensors, including the fixed filterbank buffers. `LoadSegmentationSource` continues to validate all keys/shapes/extents before payload reads and owns its weights.
- The offline exporter writes the original tensor inventory and a separate lowered `[80,251]` filter tensor. The Go constructor checks/copies this tensor; callers must verify correspondence and provenance through the manifest hashes.
- Final manifest SHA256: `bc8af5359abf662c93646fd00fe29bec02ab4e5998cef70a92be45af454ff1e0`. Two clean exports produce seven byte-identical files, including model, filters, four traces and manifest.

`scripts/community1_checkpoint_export.py` checks raw checkpoint/public PCM/source hashes before inference, uses `torch.load(weights_only=True, map_location="cpu")`, and allowlists exactly three checksum-pinned metadata types: `Problem`, `Resolution`, `Specifications`. It checks the root inventory, architecture, hyperparameters, powerset/window metadata and strict state_dict binding. There is no unrestricted unpickle or network request. The output directory must be new.

The reference extracts the pinned PyanNet/SincNet classes using the existing minimal task-independent module pattern. Four per-layer LSTMs match the full monolithic LSTM bit-for-bit; the traced manual path matches full PyanNet output bit-for-bit. It runs with one CPU thread and MKLDNN disabled. Python/PyTorch is offline conversion/reference tooling only; Go inference has no external neural fallback.

The model card declares **CC-BY-4.0** for weights, separately from the implementation source licence. Attribution and conversion changes are recorded in `models/speaker/community1/NOTICE`. The gated model, converted weights, public PCM and raw tensor traces stay in the local cache and are not shipped in this repository. The source-only manifest records their hashes and shapes.

## Runtime contract

`NewExperimentalSegmentation` builds a separate `ExperimentalSegmentation` wrapper around checked weights and lowered filters. The existing `SegmentationCheckpoint` remains feature-only. No default constructor, CLI, HTTP service or media backend changes.

`ForwardPCM` and `ForwardPCMObserved` require explicit SincNet FMA, LSTM and head modes. They consume one complete immutable mono16k window within SincNet's grid bounds, up to160000 samples, with zero recurrent initial state. They return owned frame-major log probabilities and nominal canonical-sample frame geometry. They do not resample, pad, skip silence, merge overlaps or assign global speaker identities. Bidirectional recurrence and whole-window normalization make arbitrary chunking non-equivalent.

Parameters are immutable; scratch/results are per call. Observers are synchronous, read-only transient views. Cancellation/error returns no partial result, although earlier observer calls cannot be retracted. No worker pool, driver, process or file is opened by the constructor or forward methods.

## Trained numerical results

The fixed per-boundary and final-output gate is maximum absolute error `2e-4`. Hard local speaker masks must have zero disagreeing frames. The four cases are one second of digital silence, a short synthetic wave, and the first two ten-second windows from the pinned public pyannote tutorial sample. No private audio is used.

| Case | Frames | Final log-probability max error | Hard-mask disagreements | Failing intermediate |
|---|---:|---:|---:|---|
| silence-1s | 56 | 0.00016164779663085938 | 0/56 | SincNet stage1: 1.0525763928890228 |
| wave-short | 3 | 0.0000059604644775390625 | 0/3 | none |
| public-0-10s | 589 | 0.000018596649169921875 | 0/589 | none |
| public-10-20s | 589 | 0.00001239776611328125 | 0/589 | SincNet stage0: 0.00022292137145996094 |

Three repeated normal-mode runs produce 144 intermediate/output comparisons over9,402,054 values:138 pass and the same six repeated comparisons fail. All12 endpoints and all3,711 local frame-mask comparisons pass. An additional AVX2/FMA-disabled run passes the endpoint/mask gates on the same four cases. No timing comparison was made.

`TestCommunity1TrainedSegmentation` always enforces endpoints and masks, while recording every intermediate. `GO_PHERENCE_TEST_COMMUNITY1_STRICT=1` additionally enforces boundaries and fails the silence/public-10–20s subtests. The strict Make target intentionally returns nonzero. The older synthetic four parameter-filter failures and two lowered-boundary failures are also retained.

## Boundary diagnosis

All12 isolated normalization comparisons using exact reference pooled inputs are bit-exact. Differences appear in convolution results before normalization. On the silence case, the reference's stage1 convolution varies slightly across frames for each channel (ranges about8.58e-5 to0.00299835); later normalization magnifies these differences. Matching final probabilities does not clear this intermediate error.

The public stage0 error is also a convolution-path difference. The exact upstream shape-dependent reduction/tail behaviour is not yet reproduced. No tolerance was widened, and no arbitrary silence special case was added.

## Verification

- Three exporter safety/metadata unit tests pass without model execution.
- `make speech-sincnet-fma-check speech-foundations-check speech-media-integration`: pass, including FFmpeg integration and source vet.
- Community-1 plus `backends/simd/...`:630 passing test events /273 top-level passes. Five explicit opt-in test skips plus the existing no-tests package skip are retained.
- Thirty shuffled experimental-wrapper runs:90 passing tests, no skips/failures.
- Forced AVX2/FMA-off wrapper tests:3pass. Trained normal runs:15pass events (12cases +3parent); trained fallback:5pass events. Endpoint gates are separate from strict boundary failures.
- Tests cover explicit mode validation, nil/invalid model/PCM, source/result ownership, exact composition with existing feature APIs, callback order, cancellation after every observed stage and sampled internal checkpoints, recovery, and four concurrent calls.
- Affected source vet passes. Community-1 arm64 test binary cross-builds; it was not executed on arm64.
- Whole-tree build retains the exact saved baseline errors. Race compilation still fails because `gcc` is absent. Ordinary concurrency tests are not race qualification.

No independent delegated review is claimed. This increment leaves embedding, whole-file segmentation scheduling, full/exclusive diarization, DER, Vulkan placement, and combined-job performance unfinished. The optional go-264 adapter remains isolated and unmerged; its separate frontend comparison cannot qualify this Go model.

## Reproduce

Obtain the gated model under its own terms and use the pinned offline environment. Do not put the output directory in the source checkout.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/diar-export/bin/python scripts/community1_checkpoint_export.py \
  --checkpoint /path/to/3533c8/segmentation/pytorch_model.bin \
  --public-wav /path/to/pinned/pyannote-sample.wav \
  --output-dir /path/to/new/converted-cache

export GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR=/path/to/new/converted-cache
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-segmentation-check
# Expected failure until the intermediate differences are corrected:
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-segmentation-strict
```

[Evidence](evidence.json) and [conversion manifest](manifest.json) include exact hashes, commands, pass/fail accounting and source pins. LLM and speech services stayed inactive; no push, deployment or restart occurred. Compute was coordinated under `whisper-community-checkpoint-0933`.
