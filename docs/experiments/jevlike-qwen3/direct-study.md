## Direct scoring on the RTX 3060

Qwen3-4B answers more of this small validation sample correctly than Qwen3-4B-Base,
but neither passes the option-order requirement. The instruction checkpoint
reaches 80% accuracy and uses the supplied evidence; it also assigns excessive
confidence to wrong answers. Temperature scaling helps the probabilities, not
the decisions. No configuration is promoted, and the final-test partition has
not been read.

These are 60 original validation examples, 12 per source/config, selected before
scoring by a fixed hash rule. Each checkpoint made the same 228 independent
prefills, with no admission rejections. Reversed options retain stable answer
IDs. Only NLI and CLINC have separable evidence, so QA's option-only baseline is
not called an evidence intervention. The [selection policy][policy] and
[calibration/promotion policy][cal] specify the comparison.

## Answers and their failure modes

| Normal validation task | Base correct / 12 | Instruction correct / 12 |
|---|---:|---:|
| ARC Challenge | 11 | 11 |
| ARC Easy | 8 | 10 |
| CLINC labelled 8-choice, including OOS | 4 | 11 |
| CommonsenseQA | 8 | 8 |
| MultiNLI | 5 | 8 |
| Total | 36 / 60 | 48 / 60 |

Uniform random accuracy for these candidate sets is 23.17%; always selecting the
first supplied option gets 21.67%. These are position diagnostics, not trained
semantic-majority baselines. Twelve questions per task is far too little for a
precise benchmark ranking, and public data may have appeared in pretraining.

Base falls from 60% to 48.33% when options are reversed, changing 34 of 60
answers. Instruction rises from 80% to 86.67%, but still changes 8 of 60 answers
(13.33%) -- above the predeclared 10% limit. The improvement in reversed accuracy
does not make the instability disappear. Exact floating-point ties select the
first supplied candidate; synthetic tests cover strict ties and one-ULP gaps,
but do not establish how often numerical near-ties occur on unseen inputs.

| Matched evidence task | Base normal | Base shuffled | Instruction normal | Instruction shuffled |
|---|---:|---:|---:|---:|
| CLINC, 12 examples | 33.33% | 8.33% | 91.67% | 8.33% |
| NLI, 12 examples | 41.67% | 41.67% | 66.67% | 50.00% |

Shuffled-evidence accuracy is measured against the original labels and is a
sensitivity control, not labelled accuracy for the substituted premise. NLI's
instruction gain is two examples; it clears the exploratory ten-point criterion
but needs a larger sample. Across all 60 originals, option-only accuracy is
26.67% for Base and 23.33% for Instruction.

## Confidence needs a separate fit

A different 60-example sample, drawn only from calibration with the same
per-task selection rule, produced a single temperature of **6.5666879722**.
Only normal requests contributed to fitting; the fixed search interval was
[0.05,20]. All 60 were admitted. The fitted calibration accuracy was 83.33% and
NLL 0.5002, which is an in-sample diagnostic, not a held-out result.

| Normal validation, 60 examples | Accuracy | NLL | Summed Brier | ECE |
|---|---:|---:|---:|---:|
| Base, T=1 | 60.00% | 1.2492 | 0.6337 | 0.2861 |
| Instruction, T=1 | 80.00% | 3.1725 | 0.3637 | 0.1912 |
| Instruction, frozen T=6.5667 | 80.00% | 0.6493 | 0.3145 | 0.1084 |

The report includes ten equal-width reliability bins and risk/coverage counts
with Wilson 95% error intervals. Equal confidence enters coverage together.
Those intervals describe the sample; selecting a threshold after seeing the
curve would need another evaluation. No deferral threshold is promoted here.

The reporter verifies the dataset manifest, exact partition and provenance,
request checksum, candidate order, labels, selected argmax and result count.
Changing a selection file from `validation` to `calibration` cannot authorise a
fit. Frozen calibration application checks model and dataset identity and records
the calibration-report digest. The local manifests are trusted experiment
records, not signatures against a hostile filesystem.

## Storage and arithmetic

The instruction model is `Qwen/Qwen3-4B` at
`1cfa9a7208912126459214e8b04321603b3df60c`. Base's verified weight shards were
removed only after its results, references and manifests were preserved and
checksummed in `base-swap-manifest.json`. Sidecars remain. This avoided two
concurrent 8 GB checkpoints; the downloader also rejects oversized combined
assets, partial-file accounting errors and concurrent fetches into the budget
root. A killed fetch leaves its lock and partial files for explicit inspection.

The instruction template has a different hash from Base. Independent
Transformers rendering nevertheless confirms the same supported single-user,
plain-string, no-thinking branch and exact token IDs. Repeated GPU fixtures pass
the unchanged maximum-error 0.005 / RMS 0.0002 gate: hidden maximum **0.00447845**,
RMS **0.0000419163**, including 401 tokens. Selected logits differ by at most
**0.00006485** with no changed fixture decisions. The legacy CPU long-input path
has not been promoted.

New scoring output includes the digest of the complete verified file set in
`model_id`, not just a repository/revision claim. The instruction file-set digest
is `503cf44de56884c7fdb34b86afae91e6ce122800a06883b058dbfff8fc03e746`.
Base's earlier run predates that field; its original manifest and result hashes
are retained without rewriting the raw output. Files must remain immutable while
the encoder is open; startup hashing does not protect against concurrent local
mutation of memory-mapped weights.

## Time and memory

The 60 normal instruction decisions had p50 **2.0357 s** and p95 **2.4960 s**,
against Base's **2.0173 s** and **2.5246 s**. Percentiles use nearest rank. A
separate fresh-process benchmark pinned three validation requests by minimum,
median and maximum admitted token count and repeated each five times:

| Tokens / choices | Samples | Warm p50 | Warm p95 |
|---|---:|---:|---:|
| 69 / 4 | 5 | 1.6462 s | 1.6920 s |
| 98 / 3 | 5 | 2.2142 s | 2.2756 s |
| 156 / 4 | 5 | 3.1551 s | 3.1879 s |

Five samples do not estimate tail latency well. Full asset hashing took 4.5833 s,
model loading 1.1188 s, and the first 98-token request 1.9641 s. This is a cold
process/load with a warm OS file cache -- not cold storage. Prefill dominates;
phase medians and each observation are in the [machine-readable report][results].
Profiling uses synchronised wall-clock phases rather than CUDA events and adds
one upload synchronisation.

Embedding uploads were 706,560 / 1,003,520 / 1,597,440 bytes per request; only the
10,240-byte final hidden row was downloaded for CPU F64 selected-head scoring.
Two concurrent callers are serialised on the encoder lock, not batched: results
matched independent logits exactly, total wall time was 5.0016 s and one caller
waited 1.7159 s. No prefix reuse was enabled.

Encoder allocations were 7,464,810,496 bytes, leaving 4,500,881,408 GPU bytes free.
Closing restored 12,107,251,712 free bytes. Peak process RSS was 7,342,288 KiB for
the study and 7,298,168 KiB for the benchmark; these include mapped weight pages.
The failed first launch only discovered a missing GNU `time` binary and performed
no inference. That log is retained. The serving model was restored healthy after
the completed measurements; no other GPU services were changed.

## Reproducing the bounded study

Downloads are explicit and sequential. Do not run the Base and Instruction
fetches together; only one fits the approved 12 GiB asset cap. Use the existing
pinned dataset preparation and [numerical fixture commands][numerics] first.
Then, for the currently installed instruction checkpoint:

```bash
MODEL=checkpoints/jevlike-qwen3/Qwen--Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c
DATA=checkpoints/jevlike-qwen3/pilot-v4
STUDY=checkpoints/jevlike-qwen3/direct-pilot-v1
CAL=checkpoints/jevlike-qwen3/direct-calibration-v1
# These create new directories and refuse existing output.
bun scripts/jevlike-direct-pilot.ts --dataset "$DATA" --output "$STUDY"
bun scripts/jevlike-direct-pilot.ts --dataset "$DATA" --partition calibration --output "$CAL"
go build -o bin/jevlike ./cmd/jevlike
bin/jevlike score-batch -encoder-model "$MODEL" \
  -verified-assets checkpoints/jevlike-qwen3/instruction-model-verified.json \
  -requests "$STUDY/requests.jsonl" -output "$STUDY/instruction-results.jsonl"
bin/jevlike direct-report -dataset "$DATA" -selection "$STUDY/selection.json" \
  -requests "$STUDY/requests.jsonl" -results "$STUDY/instruction-results.jsonl"
bin/jevlike score-batch -encoder-model "$MODEL" \
  -verified-assets checkpoints/jevlike-qwen3/instruction-model-verified.json \
  -requests "$CAL/requests.jsonl" -output "$CAL/instruction-results.jsonl"
bin/jevlike direct-report -dataset "$DATA" -selection "$CAL/selection.json" \
  -requests "$CAL/requests.jsonl" -results "$CAL/instruction-results.jsonl" \
  -fit-temperature > "$CAL/instruction-report.json"
bin/jevlike direct-report -dataset "$DATA" -selection "$STUDY/selection.json" \
  -requests "$STUDY/requests.jsonl" -results "$STUDY/instruction-results.jsonl" \
  -calibration "$CAL/instruction-report.json"
bin/jevlike direct-bench -encoder-model "$MODEL" \
  -verified-assets checkpoints/jevlike-qwen3/instruction-model-verified.json \
  -dataset "$DATA" -selection "$STUDY/selection.json" \
  -requests "$STUDY/requests.jsonl" -repeats 5
```

Free the GPU from the authorised serving workload before execution and restore
its previous state afterward. Hashes in the report identify local raw results,
references and logs; dataset text and weight files stay outside Git.

This completes the bounded direct comparison, not issue #2. A frozen-head
comparison is now motivated by the order failures and seconds-long prefill, but
immutable feature caches, resumable head training, three seeds and fresh
human-reviewed evaluation have not been implemented or run in this stage.
Exact-prefix reuse also needs its own branch-isolation test before any speedup
claim. Final-test evaluation waits for that comparison and a frozen policy.

[policy]: direct-study-policy.md
[cal]: direct-calibration-policy.md
[numerics]: compact-direct-validation.md
[results]: direct-study-results.json
