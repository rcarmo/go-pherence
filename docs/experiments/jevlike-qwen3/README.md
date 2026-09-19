# Frozen Qwen3 choice-scorer experiment

Issue [#2](https://github.com/rcarmo/go-pherence/issues/2) is in progress. The local
RTX 3060 host was explicitly approved on 2026-09-19. Pinned data preparation and an experimental compact GPU/direct-token path are
implemented. Numerical fixtures pass for the downloaded Base checkpoint; this
is not a trained-model quality result or an instruction-following claim.

## Data and model identity

[sources.json](sources.json) pins full Hugging Face revisions, file lengths,
LFS SHA256 values and Git blob identities. `scripts/jevlike-fetch.ts` is a dry run
unless `--download` is supplied. Downloads are resumable, verified before rename,
and refuse the 12 GiB asset budget or less than 30 GiB remaining disk. Actual
SHA256 records for all files stay beside the ignored downloads.

The Qwen3-4B-Base revision is
`906bfd4b4dc7f14ee4320094d8b41684abff8539`: 11 selected files, 8,056,517,036 bytes,
including 4,022,468,096 BF16 parameters. The config has width 2560, 36 layers,
32 attention heads, 8 KV heads, head width 128 and vocabulary 151936.

The ordinary dense GPU loader expands tensor data into F32 device buffers and
allocates generation resources. The separate experimental `FrozenGPUEncoder`
keeps BF16 transformer matrices resident, widens/transposes one matrix into
reusable F32 scratch, and returns either all final-normalised rows or the last
row for selected-token scoring. It allocates no vocabulary projection buffer.
See [compact/direct validation](compact-direct-validation.md) for limits,
reference comparisons, cleanup and the explicitly reported CPU selected-head
projection.

## Licences and pilot policy

| Source | Card terms | Pilot handling |
|---|---|---|
| MultiNLI | Mixed OANC/source, CC BY 3.0, CC BY-SA 3.0, MIT and other terms | Fiction excluded; retain the pinned card/source terms. This is not an unrestricted dataset claim. |
| CommonsenseQA | MIT as recorded by the card | Supplied distractors retained; ambiguous duplicate answer text rejected and recorded. Public unlabelled test is not downloaded or used. |
| AI2 ARC | CC BY-SA 4.0 | Easy and Challenge preserved as separate source configurations; original alternatives retained. |
| CLINC OOS, `plus` | CC BY 3.0 | Pinned 151-label vocabulary, readable underscore-expanded labels, 8-choice candidate sets with gold and OOS included. |

Dataset text, cached features and trained artefacts remain outside Git. The
repository licence does not replace source dataset terms; redistribution needs
separate review. CLINC candidate construction is a labelled candidate-set task,
not 151-way deployment accuracy: wrong alternatives favour label-word overlap,
with deterministic tie breaking. Exact choices are preserved in provenance.
OOS is an intent label, not a missing-answer or confidence-deferral label.

The original pilot selection targets 1,200 training originals from each family
(ARC: 600 Easy + 600 Challenge), then carves training groups approximately
80/10/10 into train/validation/calibration. Official held-out inputs become
final-test only. MultiNLI selection is genre-stratified and keeps prompt groups
whole; CLINC is intent-stratified. MultiNLI label balance is recorded rather than
forced by splitting prompt groups. No permutations or augmentation precede the
split. Unknown boundaries, contradictory labels, duplicate choices and
cross-boundary groups fail preparation.

One pinned-source defect was observed: MultiNLI reuses some `pairID` values for
different hypotheses. The conversion export preserves `original_pairID` and
adds a content-hash suffix; `promptID` remains the grouping key. CommonsenseQA
has 122 training rows with invalid/ambiguous choice sets; excluded IDs and reasons
are retained in `selection.json`, not silently relabelled.

The first accepted export has 6,240 originals: 4,800 from training sources and
1,440 held-out. One exact duplicate is removed. Prepared counts are **3,839 train,
478 validation, 482 calibration and 1,440 test**. Two independent preparation
runs produced identical manifests and split hashes. These counts precede the
future tokenizer-aware overlength gate; no data may be silently truncated while
retaining its label.

## Reproduction

All commands run from the repository root. Python is conversion-only; task
adaptation, strict validation, grouping, deduplication and permutation are Go.
The small conversion environment needs `pyarrow==21.0.0`.

```bash
bun scripts/jevlike-fetch.ts --kind datasets --download
bun scripts/jevlike-fetch.ts --kind model --download

.venv-dataset/bin/python scripts/jevlike-export-parquet.py \
  --output checkpoints/jevlike-qwen3/pilot-exports-v3

go run ./cmd/jevlike prepare \
  -manifest checkpoints/jevlike-qwen3/pilot-exports-v3/inputs.json \
  -output-dir checkpoints/jevlike-qwen3/pilot-v4
```

Outputs must not exist. `prepare` checks revision/licence fields and exact input
hashes, writes four plain ChoiceExample JSONLs, a lineage sidecar and hashed
manifest, and publishes the directory only after successful preparation.
Duplicate rows retain sidecar lineage, and each provenance row records its
1-based output row for per-task evaluation without putting metadata into model
input. No backbone is loaded at this stage.

```bash
go test ./model/jevlike ./cmd/jevlike
go vet ./model/jevlike ./cmd/jevlike
go test -race ./model/jevlike ./cmd/jevlike
bun test scripts/jevlike-fetch.test.ts
python3 scripts/jevlike-export-parquet.test.py
```

## Next gate

Per the [direct-logit-first update](https://github.com/rcarmo/go-pherence/issues/2#issuecomment-5741524821),
run the direct scorer as a validation baseline before feature caching or head
training. The prepared final test remains excluded from iterative selection.
Instruction-tuned Qwen3 requires sequential asset use or an approved budget
revision; another 8 GB checkpoint must not be added under the 12 GiB cap.

The planned cache stays capped at 12 GiB. Calibration, quality/control comparisons,
three-seed head comparisons and human-reviewed new evaluation remain pending.
Numerical parity is not model quality; do not close issue #2 on this milestone.
