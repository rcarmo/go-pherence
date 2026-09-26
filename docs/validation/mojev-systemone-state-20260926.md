# MoJev structured-state input parity for JevBench

`DecodeSystemOneTextRequest` adds a separately named compatibility path for
object, array and null states. `DecodeTextRequest` remains text-only. This closes
the35 structured-state refusals in the diagnostic4096-capacity JevBench run;
full inference results must still be measured, not inferred from parsing.

## Contract

The pinned GSO `entryText` renderer keeps text unchanged and applies compact JSON
to objects, arrays and null. Member order, number spelling and string escapes
are preserved. MoJev's compatibility decoder follows that rule, independently
implemented; it does not reuse Simple-JEV's sorted-key serializer or MoJev
upstream's whitespace-bearing Python JSON rendering.

MoJev retains bounded/safe input requirements:1MiB request,64KiB structured state,
maximum32 container levels, unique decoded object keys and reserved-token checks.
The private structured-state marker causes `ValidateTextControls` to validate
decoded keys and values as well as rendered JSON. Thus `\u003c|reserved|>` cannot
bypass controls through escaping. Revalidating the current state also protects
against caller mutation after decoding. Plain-text input retains its existing
decoded-length policy, including long escaped representations.

This is **not full GSO wire parity**. Structured instructions/criteria remain
unsupported, and malformed JSON, duplicate keys or oversized/deep states are
stricter than GSO. Every public JevBench item fits this explicitly tested subset.
Model-specific prompts and tokenizers remain distinct; identical input semantics
do not imply identical prompt token IDs or matching predictions.

## Independent reference and corpus gate

GSO's source at historical benchmark revision
`b18ee0d4748bac436999aa72c000e06406c3cce6` matches current `systemone.go` byte-for-byte,
SHA-256 `1cd60986f1a3e37d07f7743e9e126d26a96f54047527836062b453e410b7fde8`.
`scripts/mojev_gso_state_oracle.ts` invokes the actual unmodified `entryText`
through a Go test overlay. Neither GSO's checkout nor production service changes.

Two model-free generations yield byte-identical `testdata/gso_state_render.json`
(SHA-256 `11c2dbe0a1a4038aab68c0059ec6ab826dbbb529afbe07a0c101ed46ebc30f43`).
Ten observations cover text, object, array, null, Unicode/escapes, lexical numbers,
empty containers and invalid scalar/trailing values. Unit tests pin the fixture
and generator hashes, compare exact strings and preserve the strict decoder.

An additional model-free preflight compiles **all231 archived GSO requests** with
its actual compiler and compares each rendered state to MoJev. All231 match
exactly, pass control/question validation, and fit qualified CPU token limits.
Maximum packed length is4065 and maximum branch length3930. The exact benchmark
revision remains `2fa63fa3226cb369795525ed011800f57dcbd894`.

The first GSO test overlay used a symlinked repository path and executed no
matching tests; that log is retained and not a pass. Canonicalizing the source
path fixes it, and the script now requires the output fixture to exist. A
preflight compile failure caused by wrong local helper names is likewise kept;
the corrected preflight completed both actual package tests.

## Gates and limitations

Focused tests/races count10 cover rendering, transactional rejection, duplicate
keys, byte/depth limits, escaped marker keys/values, caller mutation and concurrent
decoding. Whole-tree CPU race passed (wall5m01.52s before the final escaped-text
compatibility refinement); final focused checks cover that refinement. Vet,
build, docs checks and ARM64/RISC-V cross-builds pass. No GPU was executed.

A delegated judge review of the explicit design found no blocker for the tested
corpus, with the scope/safety divergences above kept explicit. It was not an
independent filesystem audit. Statement coverage is reported in the evidence;
JSON decoder's unreachable-after-validation branches prevent claiming complete
coverage. No tokenizer/model arithmetic or quality threshold changed.

Evidence: `/workspace/tmp/mojev-systemone-parity-20260926`, including generated
oracle tests/overlays, corpus preflight output, failure logs and correctness gates.
The official JevBench sealed score remains unavailable. Fresh full-corpus
inference is separate from this model-free compatibility qualification.

```sh
bun scripts/mojev_gso_state_oracle.ts /path/to/go-system-one /tmp/state-oracle
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/mojev \
  -run 'Test(SystemOneState|TextControls|PinnedMoJevTextRequests|DecodeText)' \
  -count=10
```
