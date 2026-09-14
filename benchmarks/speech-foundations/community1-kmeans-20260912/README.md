# Community-1 forced-count K-means — 12 September 2026

This model-free checkpoint implements the bounded forced-speaker-count fallback used by pinned Community-1 postprocessing when multirow VBx output falls outside the requested speaker-count range.

Baseline: `6a87da48aeba524b4af27c5d8c941c2292ea0c59` on `feat/speech-simd-vulkan`.

## Contract

- Input embeddings remain caller-owned float32 rows. K-means runs on cosine-normalised rows with the scikit-learn 1.9.0 defaults used by pinned pyannote: Lloyd iterations, greedy k-means++, `n_init=3`, `RandomState(42)`, `max_iter=300` and the source tolerance scale.
- NumPy legacy MT19937 output and 53-bit `random_sample` construction are implemented locally. The runtime has no Python, NumPy or scikit-learn dependency.
- Geometry is capped at 512 rows, 512 dimensions and 64 clusters. A conservative `3*300*rows*clusters*dimensions <= 2^28` check rejects excessive synchronous work before allocation.
- One admitted training row retains the existing source shortcut and can report an unmet count. Only multirow VBx count mismatches use K-means.
- `NumSpeakers` overrides min/max. A VBx count below the effective minimum targets the minimum; a count above the effective maximum targets the maximum.
- Forced centroids are recomputed from the original float32 embeddings after widening values to float64. Constrained assignment is disabled, matching the source.
- Results report `Path="clustered-kmeans"`; `InitialLabels` retain AHC labels and `KMeansLabels` expose forced training labels. The durable schema accepts the new path and uses stage identity `speechjob-community1-rawturns-source-timing-kmeans-v3`.
- Cancellation and observer errors return no partial result. Returned labels and centroids own their storage.

## Empty-cluster portability

Pinned scikit-learn relocates empty Lloyd clusters with NumPy float32 `argpartition`. NumPy 2.5.3 dispatches that selector to ISA-specific x86-simd-sort implementations; selected donor order can differ even without equal distances, and equal-distance identities are also unstable.

The Go path therefore preserves portable source behavior only where identity is defined:

- one empty cluster with one unique farthest donor uses that donor;
- duplicate-only zero-distance empties keep source behavior by copying the largest cluster center;
- multiple empty clusters or equal farthest donors return `ErrAmbiguousKMeansRelocation` rather than inventing a portable speaker identity.

Direct low-level tests match pinned Lloyd outputs for the unique-relocation and duplicate-only cases and verify the explicit ambiguous failure. This does not claim cross-ISA NumPy identity where none exists.

## Reference provenance

- NumPy `2.5.3`; mtrand binary SHA256 `53a0dea25039fd6e55ce2a6c94789c9273fdfa1cc4bda60dde6e813f455c9bc5`.
- NumPy 2.5.3 sdist SHA256 `df2d5874ff183595a4ba404edd04f6bd9b5505c1d7708573f6a6c17489a67563`.
- NumPy `numpy/_core/src/npysort/selection.cpp` SHA256 `228a1bb4b2c47b66855548bdf432337efdfbd7cc27cd7d752d52f25543c227b3`.
- scikit-learn `1.9.0`; `_kmeans.py` SHA256 `7d9cd3c75f1c40616223fbceb23bc1e115de043b355ab90716898301756d746c`.
- scikit-learn `_k_means_common.pyx` SHA256 `4b00b0ba8d8317847489c8cae283258e15b8bf7d0ea7067ad06b2a1bde0340c4`.
- scikit-learn `_k_means_lloyd.pyx` SHA256 `8aa6c66d2ea790d420c4da001e44c75f941a69c45c5953f9989ca7b1ed99f894`.
- Installed common/Lloyd extension hashes are pinned by the standalone generator as `28273c46e0c190277ad7739aadc086c7dacf6a9ad9e0232a26f223b67274b25e` and `888274137c2f633b5814926ebdea181b7c508222549d0ad7e0c5c4554e831214`.
- Separate scikit-learn and NumPy BSD-3-Clause terms are retained beside the Community-1 package.

Both Python generators fail closed on pinned source/version drift and perform no model, audio, network or GPU work.

## Fixtures

- `models/speaker/community1/testdata/kmeans-reference.json`: 27 synthetic float32 cases plus 24 exact MT19937 doubles; SHA256 `c193ce9ca94026b4d3c92a66bed8eb6554cd4915eb90f934094557b77cfcff34`.
- `models/speaker/community1/testdata/postprocess-reference.json`: schema 2, 11 connected source-extracted cases including forced one- and three-cluster paths; SHA256 `2fdbbf6f82d7202dc615c5d4af5b6163c1737c8f04403a6d13833e7e94a1c22b`.
- Each generator was run twice; each output was byte-identical on regeneration.

## Verification

Toolchain: Go 1.26.2, Linux/amd64. `GO_PHERENCE_DISABLE_NVIDIA=1`; server/profile checks use the documented `GOMAXPROCS=2`.

Passing checks after the final relocation correction:

- `go test -p=1 -count=1 -timeout=600s ./models/speaker/community1 ./runtime/speechjob`
- `go test -p=1 -count=10 -shuffle=on -timeout=1200s ./models/speaker/community1 ./runtime/speechjob`
- `go test -p=1 -count=1 -timeout=300s ./runtime/speechjob/httpapi ./cmd/audio/speechjobserve`
- `go test -p=1 -count=3 -shuffle=on -timeout=600s ./cmd/audio/speechjobserve`
- `go vet ./models/speaker/community1 ./runtime/speechjob ./runtime/speechjob/httpapi ./cmd/audio/speechjobserve`
- `go build ./models/speaker/community1 ./runtime/speechjob ./runtime/speechjob/httpapi ./cmd/audio/speechjobserve`
- Linux/arm64 `go test -c` for `models/speaker/community1`, `runtime/speechjob` and `runtime/speechjob/httpapi`.
- `gofmt`, `git diff --check`, no `scripts/__pycache__`, and no temporary SincNet probe.

The server command is intentionally Linux/amd64-only because its Community-1 profile API is build-tagged `linux && amd64`; it is not an arm64 acceptance target. Its full native package and three shuffled repetitions pass.

## Baselines and exclusions

- `make speech-foundations-check` could not start because `make` is absent from the container. The equivalent affected Go tests/vet/builds above passed.
- `go build ./...` still fails in unrelated pre-existing SpacemiT AICPU and DiffusionGemma command packages due missing symbols/fields. No failure names the changed Community-1 or speech-job packages.
- `go test -race` could not start because CGO is disabled in this environment. It is not counted as a pass.
- No trained model, native GPU, corpus, private audio, service, deployment, push, pin, default, tolerance or quality gate changed.
- Strict SincNet qualification, trained/public-corpus quality, native placement and whole-job performance remain open.
