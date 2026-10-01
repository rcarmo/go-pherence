# Experimental original key-extent compatibility — 1 October 2026

Virtual zero-key padding reproduces physical padding bit-for-bit and removes the known JFK VAD `Thank you.` tail. It does not establish original timestamp/word-quality parity or a speed gain. The new mode is explicit and experimental; defaults, GELU, decoder policy and services remain unchanged.

## Retained mode

`NewVkAttentionKey32PaddedExtentF32` includes keys through `ceil(seqKV/256)*256`. Actual K/V loads remain guarded by `seqKV`; virtual positions have zero key and value, but participate in softmax. Query/output extents are unchanged. For the turbo encoder,1500 actual keys become1536 logical keys. These unmasked zero positions contribute to the denominator, matching the original attention boundary identified in [common-state attribution](whisper-common-state-20261001.md).

`NewVulkanEncoderOriginalQ5PaddedKeyExtent` and benchmark backend `vulkan-original-q5-padded-keyextent` select this operator with original-Q5 decode4 FFN and retained erf GELU. Other modes keep their previous operators. The public constructor requires the same checked original source; the packed-only benchmark uses verified metadata and original blocks. No activation requantisation, F16 storage, audio padding/context changes, text filtering or automatic fallback occurs.

The operator requires headDim64. The existing sequence/head/device envelopes remain unchanged. K/V allocation size stays at the actual extent; shader groups, four buffers,20-byte push constants,16×16 local geometry and14,528 shared bytes match outputILP. Complete encoder stats remain34 plans/390 stages,1,184,890,880 weight bytes and93,696,000 scratch bytes. No physical padding buffer or copy stage is added. The closed regeneration inventory increases from31 to32 shaders without widening instruction/feature admission.

This changes attention semantics relative to the accepted true-key mode. It is not an exact-output optimisation of that mode, and the original's F16 storage/GELU/other arithmetic differences remain separate.

## Native and model-free checks

Ten native shapes compare the virtual operator with the ordinary outputILP operator fed physically appended zero K/V rows:1/1/1,17/37/2,33/65/3,16/255/2,16/256/2,16/257/2,1500/1500/20,1/4095/1,4096/1/1 and2/33/32 (queries/keys/heads). All output bits match across repeated execution, including output/query tails,256-key boundaries and wholly virtual tiles. Pre-cancellation, fresh reuse, allocation guards and cleanup pass. Selected small shapes also pass the existing float64 analytic budget of2e-5 absolute/relative; no tolerance is widened.

The full pinned turbo encoder compares virtual padding with a separate test-only physical-copy oracle. All1,920,000 hidden values match by F32 bits. Three cancellation checkpoints, native drain where required, exact fresh reuse and complete cleanup pass on the final tree;1243 checkpoints are recorded. Model stats match the physical oracle's encoder stats; external oracle scratch/copies are deliberately outside those stats.

Offline tests cover shader ABI, cancelled construction, required head dimension, shared-memory limits and unchanged default selection. Model admission tests reject nil/cancelled/unverified-source requests. These tests establish the tested operator/ownership contract, not independent acoustic or long-form correctness.

## Speech results: five repeats per fixture

Every candidate repeats its own complete output exactly across five requests. Inputs, original-model pins, language, CPU4/heap4GiB, native VAD, word flags and gap policy are retained. CPU parallel-row scheduling is enabled for baseline and candidate.

| Fixture | Observed change from true-key Go output |
|---|---|
| JFK noVAD | Same text/content tokens and0–10.40s segment |
| PT row0 noVAD | Same text/tokens; endpoint7.36→7.38s |
| FR row0 noVAD | Same text/tokens and0–2.42s segment |
| JFK VAD + words | Removes `Thank you.`;2→1 segments and24→22 words;2 shared speech-word boundaries shift20ms |
| Two JFK groups, VAD + words | Removes both extra tails;4→2 segments and48→44 words;2 shared speech-word boundaries shift20ms in each group |
| Extra PT row1 | Complete returned output unchanged |
| Extra PT row2,43.76s | Same content tokens; first boundary4.98→4.96s |
| Podcast20s noVAD | Same content tokens; several segment boundaries shift20ms |
| Podcast20s VAD + words | Same content tokens/78 words;6→7 segments;6 word boundaries shift20ms |
| Silence VAD + words | No transcript windows |

The JFK result confirms the known tail can disappear without a heuristic text filter. The word/segment shifts require independent qualification. Exact old-path word-output parity fails; it is not silently relaxed.

## Original output comparisons

Pinned original whisper.cpp Vulkan/flash requests on the same canonical PT2/podcast PCM execute five repeats with stable outputs. Original DTW is disabled under flash; the VAD comparison also uses original compaction rather than Go gap preservation. These runs compare transcription/segment output, not matched word alignment or acoustic truth.

PT row0's retained original endpoint is7.36s, so the new7.38s endpoint misses that timestamp. The candidate PT2 first boundary4.96s matches the fresh original, but its last `E aí` segment still ends43.76s versus original35.12s. Both candidate and old Go share that endpoint difference. Podcast content is unchanged but original noVAD uses five segments versus candidate six; VAD segmentation/timing also differs. No general original-output or independent quality acceptance follows.

## Request cost and noisy pairs

Initial candidate processes precede the baseline processes; five repeats include the first. They are sequential pairs with no randomisation, clock control or discarded samples. Output changes can alter decoder/alignment work, so these costs cannot all be interpreted as equivalent-work kernel gains.

| Fixture | Initial true-key median (s) | Padded median (s) | Raw change |
|---|---:|---:|---:|
| JFK noVAD |6.569 |6.144 |−6.47% |
| PT noVAD |6.215 |6.104 |−1.78% |
| FR noVAD |5.652 |5.704 |+0.91% |
| JFK VAD/words |7.022 |6.854 |−2.40% |
| Two groups VAD/words |14.156 |13.861 |−2.09% |

The JFK baseline was unusually slow. A later same-executable five-repeat baseline→candidate pair gives6.097→6.135s (+0.61%) with identical output. The initial apparent6.47% gain is excluded from accepted speed claims; both raw runs remain archived. VAD/word arms return fewer tokens/words with padding and are not matched-output workloads. Expanded fixtures use earlier retained Go baselines only as output references, not speed pairs. Setup and full-arm totals remain separate in JSON. No original-speed gain or regression is qualified.

## Verification and status

The final tree passes `make model-layout-check host-build host-vet host-test docs-check`, whole-tree race, marked ARM64/RISC-V builds and final native operator/full-encoder gates. All32 stored/embedded shaders validate and regenerate with normalised identity. Final accepted-decoder-opt-in race and documentation checks are recorded after this report. Defaults remain the previous serial decoder/true-key encoder selections.

Setup failures were corrected before successful gates: the shader generator initially missed a whitespace anchor; the constructor used the wrong private helper name; a benchmark admission edit omitted `opts.`; the inventory test initially lacked the new embedded source. These failures are not native passes or quality results. The final shader comment describes zero-only tiles correctly, and its stored/embedded SPIR-V is regenerated and rechecked.

The read-only source-review delegate timed out after60s; no source-review approval is claimed. A supplied-facts judge completed and supports retaining an explicit experimental path with the stated limits. It did not audit raw/source artifacts or provide acoustic validation.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-padded-keyextent-20261001/) retains raw baseline/candidate/expanded/original outputs, rerun timings, environment settings, physical-padding oracle source/SPIR-V, native states/exits/monitors, parser/comparison, tool/binary provenance and full gates. No audio, weights or executable binaries are committed.

The fresh @llama hold excludes competing experiments/builds/restarts. All native deadlines stay≤120s under CPU4/8GiB/no-swap, physical Intel GPU, Qwen-idle and host available memory≥6GiB guards. Successful native/build containers exit0/noOOM, are removed and actually drain before release. Qwen LAN, Gemma, services and resource allocations stay unchanged. No deployment occurs.

Original-compatible activation/storage arithmetic is not combined into this mode. Independent acoustic/word timing, long-form/resume, device-fault qualification and the overall original-speed target remain open. This option is retained for those further comparisons only.
