# Smallest Pocket full-step boundary proof

The smallest missing proof is a computed row-to-later-graph join against the existing full-step PyTorch fixture. The fixture already contains all 59 trainable gradients; the accepted grouped harness currently proves separate components and a later graph whose row boundaries are supplied.

This is read-only planning at `a90aec30567d485784f2e04d1803c543c9f2817f`, owned by `@go-pherence`. Authorisation: `/workspace/exports/pocket-local-grouped-alias-correction-decision-20261002.md`. No code, fixture generator, test or graph was run or changed for this plan. Only this planning note was added. The accepted adapter/coordinator, runtime seals and inference index remain unchanged. Return this proposed slice to `@default` before implementation or numerical gates.

## Existing independent evidence

`testdata/training_step_pytorch.json` has schema 1, upstream revision `0acce6b2f390150267557770d2098c5caa9a18ac`, generator `pocket-tts .venv/bin/python (torch 2.13.0+cu130)`, seed 1234. Static JSON inspection gives:

- Frames 3, voice frames 2, hidden 4, latent 2, vocabulary 6; text IDs `[1,3]`; mask `[true,true,false]`; two valid flow rows.
- `PEqual=.75`, `EOSLossWeight=.1`, flow batch multiplier 1. Thus diagonal objective scale is `.375`, distillation scale `.125`.
- Input latents/noise: six values each; voice: four. Time arrays each have three values, padded final time zero. Diagonal times `.2,.7`; distillation `(s,t)=(.15,.8),(.3,.65)` on the valid rows.
- Final five metrics, two valid-row logvar outputs and their derivatives per branch, six final audio derivatives, four voice derivatives, and 59 named parameter gradients / 606 elements.
- Four nontrainable state arrays: `flow_lm.emb_mean`, `flow_lm.emb_std`, `flow_lm.flow_net.time_embed.0.freqs`, `flow_lm.flow_net.time_embed.1.freqs`. They are frozen sources, never G roots. AdamW/EMA fields exist but are outside this proposal.

`generate_training_step_pytorch.py` calls the pinned upstream `TrainableTTS.forward` and `loss.backward()` with deterministic LSD time sampling and recorded noise. It exports the final derivatives directly from upstream autograd. It does not export per-row hidden/condition VJPs, direct-target VJPs, raw branch statistics, endpoint input VJPs or per-row flow-G decomposition.

`TestPocketTrainingStepPyTorchParity` in `training_step_parity_test.go` compares the CPU full step with this fixture at metrics `3e-5`, logvar/audio/voice derivatives `5e-5`, every named G `8e-5`, and exact zero padded logvar **derivative** rows. It is an existing complete **CPU** proof, not a grouped adapter proof. Expected values must remain the fixture arrays; neither `PocketTrainingStep` nor the accepted adapter may generate new golden arrays.

`flow_head_backward_pytorch.json/lsd_minimal` separately supplies a pinned minimal-LSD row's velocity, tangent, endpoint, residual, loss, raw square, condition/target/noise/time/logvar derivatives and all 34 flow gradients. `TestLSDDistillMinimalPyTorchParity` keeps loss/square `2e-6`, forward arrays `5e-6`, row derivatives `2e-5`, parameters `3e-5`. Its geometry/condition/weights differ from the complete step fixture; it cannot be spliced in as a row of that step.

## Row-to-later boundary map

Notation: `z_r` is the audio hidden row, `y_r` the normalised target, `n_r` frozen sampled noise, `C=2`, `V=2`, `a=p/V`, `b=(1-p)/V`. All arrays are frame-major. Inactive rows remain explicit zero derivatives/metrics; they are never omitted from shape admission.

| Phase / source | Computed outputs and scale contract | Exact consumer |
|---|---|---|
| Weighting forward, `BuildWeightTrainingTape` | Pack diagonal `(time,time)` rows then distillation `(s,t)` rows; affine/ReLU logvars, P fixed | Valid row logvars feed both early losses. Compare valid prefixes to fixture; padded logvar **outputs** need not be zero, but their seeds must be zero |
| Backbone forward, `BuildBackboneTrainingTape` with zero seed | Sequence and transformed full prefix+audio. Rows=8, prefix=5; audio hidden is final 3×4 suffix | `z_r` supplies flow condition and EOS overflow admission; final combined graph recomputes the same backbone from frozen P/batch |
| EOS projection forward, `BuildEOSProjectionTrainingTape` | Unweighted masked BCE mean. Admission weight `.1`; mean itself is not preweighted | `StepReductionInputs.EOS`; final loss adds `.1*EOS`. Forward-only execution commits no EOS G |
| Diagonal forward, `BuildDiagonalForwardTape` | `x=time*y+(1-time)*n`, desired `y-n`, prediction at `(time,time)` | Complete prediction/desired overflow admission then `BuildDiagonalTrainingTape` |
| Diagonal backward | Loss stats `[rawSquare, loss, dLogvarRaw]`; prediction seed multiplied by `a` **inside** graph. Returns scaled condition derivative and scaled direct target `time*dX - scaledSeed`; all 34 G already include `a` | `DiagonalCondition`, `DiagonalTarget` copied without another scale; raw stats/logvar feed reduction. Row G adds with unit accumulation |
| Distillation forward, `BuildDistillForwardTape` | `xs=s*y+(1-s)*n`; primary velocity `v`, selected end-time JVP `q`; `delta=t-s`; endpoint input `xt=xs+delta*v`; desired `v+delta*q`; endpoint prediction at `(t,t)` | All six owned roots plus condition/target/noise/time/logvar feed `BuildDistillTrainingTape`; preserve separate overflow admission |
| Endpoint reverse, inside distillation | Residual desired−endpoint, unscaled seed `dr`; endpoint reverse seed `−dr`; only `dEndpointInput` returns. `GV=dr+delta*dEndpointInput`, `GT=delta*dr`; delta adjoint is dot(dr,q)+dot(dEndpointInput,v) | Primary mixed reverse receives GV/GT and endpoint-input/delta edges. Endpoint P/condition/time gradients never enter total G |
| Primary mixed reverse/finish | Raw condition VJP, direct target `s*(dPrimaryInput+dEndpointInput)`, noise and time adjoints; `[rawSquare,loss,dLogvarRaw]`; 34 primary-only raw flow G | `DistillCondition/Target/Logvar/RowMetrics` receive raw values. Flow G adds with `b`, exactly once; reduction separately scales the raw input/logvar boundaries |
| Reduction, `BuildStepReductionTrainingTape` | `DCondition=diagScaled+b*distRaw`; `DTarget=diagScaled+b*distRaw`; logvar seeds `a*diagRaw`, `b*distRaw`; metrics columns averaged over V | Weight MLP backward uses scaled logvar IDs; backbone uses combined condition; final input uses direct target |
| Weight join, `BuildStepWeightTrainingTape` | Samples' raw seed arrays must exactly equal reduction raw logvars; remap to scaled reduction seeds before MLP VJP | Six `flow.w_s_t.*` G, accumulated once with unit scale; no second `a/b` |
| EOS/backbone join, `BuildStepBackboneTrainingTape` | Zero-pad combined condition seed to five prefix rows; recompute hidden; EOS VJP uses masked `.1/V` once; combine EOS hidden seed with flow condition inside backbone | 19 FlowLM G including both EOS arrays, voice VJP, shifted-audio VJP |
| Final selected results, `buildStepResultsTape` | Add direct target once to backbone audio VJP; logvars/metrics/optional voice/normalised/tail roots | Real `decodeStepResultsBoundary`; latent-sized direct tail equals final normalised row and is zero for the padded row |

`training_step.go` explicitly places these fields into `StepReductionInputs`. `training_step_reduction_tape.go` outputs `{dz,target,dd,dt,metrics}`, with metrics ordered raw diagonal square, raw distillation square, weighted flow loss, unweighted EOS mean, total loss. `training_step_weight_tape.go` remaps `sampleSeedIDs` to `reduction.Outputs[2+i]`. `training_step_backbone_tape.go` pads the reduced condition seed, appends EOS/backbone and adds `step.Outputs[1]` to normalised-audio VJP.

The public diagonal/distillation backward builders accept earlier forward arrays for admission/explicit edges and recompute their immutable flow graph. These arrays must come from the same frozen P and forward inputs. A shape match alone does not establish provenance. No use of the production retained-device owners is needed for a host proof.

## Exact total G inventory

The complete fixture partitions into disjoint name families: flow **34 / 386 elements**, FlowLM including EOS **19 / 195**, weighting **6 / 25**; total **59 / 606**. Each phase must assert exact expected names, unique P/G IDs, declared root occurrence, finite values and extents before comparing. No count-only acceptance.

The table below is the exact full-step fixture inventory. Brace patterns expand to every listed index; no other index or name is admitted.

| Prefix / exact suffix pattern | Indices | Elements per named tensor | Names |
|---|---|---:|---:|
| `flow_lm.flow_net.cond_embed.bias`, `.weight` | — | 4, 16 | 2 |
| `flow_lm.flow_net.input_proj.bias`, `.weight` | — | 4, 8 | 2 |
| `flow_lm.flow_net.final_layer.adaLN_modulation.1.bias`, `.weight` | — | 8, 32 | 2 |
| `flow_lm.flow_net.final_layer.linear.bias`, `.weight` | — | 2, 8 | 2 |
| `flow_lm.flow_net.res_blocks.{i}.adaLN_modulation.1.bias`, `.weight` | i=0,1 | 12, 48 | 4 |
| `flow_lm.flow_net.res_blocks.{i}.in_ln.bias`, `.weight` | i=0,1 | 4, 4 | 4 |
| `flow_lm.flow_net.res_blocks.{i}.mlp.{j}.bias`, `.weight` | i=0,1; j=0,2 | 4, 16 | 8 |
| `flow_lm.flow_net.time_embed.{i}.mlp.{j}.bias`, `.weight` | i=0,1; j=0,2 | 4, 16 | 8 |
| `flow_lm.flow_net.time_embed.{i}.mlp.3.alpha` | i=0,1 | 4 | 2 |
| `flow_lm.bos_before_voice` | — | 4 | 1 |
| `flow_lm.bos_emb` | — | 2 | 1 |
| `flow_lm.conditioner.embed.weight` | — | 24 | 1 |
| `flow_lm.input_linear.weight` | — | 8 | 1 |
| `flow_lm.speaker_proj_weight` | — | 8 | 1 |
| `flow_lm.out_eos.bias`, `.weight` | — | 1, 4 | 2 |
| `flow_lm.out_norm.bias`, `.weight` | — | 4, 4 | 2 |
| `flow_lm.transformer.layers.0.layer_scale_{i}.scale` | i=1,2 | 4 | 2 |
| `flow_lm.transformer.layers.0.linear1.weight` | — | 24 | 1 |
| `flow_lm.transformer.layers.0.linear2.weight` | — | 24 | 1 |
| `flow_lm.transformer.layers.0.norm{i}.bias`, `.weight` | i=1,2 | 4, 4 | 4 |
| `flow_lm.transformer.layers.0.self_attn.in_proj.weight` | — | 48 | 1 |
| `flow_lm.transformer.layers.0.self_attn.out_proj.weight` | — | 16 | 1 |
| `flow.w_s_t.0.bias`, `.weight` | — | 3, 6 | 2 |
| `flow.w_s_t.2.bias`, `.weight` | — | 3, 9 | 2 |
| `flow.w_s_t.4.bias`, `.weight` | — | 1, 3 | 2 |

The full-step transformer linear1/linear2 and attention projections have no bias entries. Expand only this fixture's exact JSON names; do not infer trainable names from another architecture. The component fixtures' same family counts do not guarantee identical extents: full-step flow has 386 elements versus the accepted mixed-flow fixture's 382.

Flow total order is valid row 0 diagonal then distillation, followed by row 1 diagonal then distillation, matching `training_step.go`. For each name, proposed total is the sum of both already-scaled diagonal row G plus `b` times each raw primary distillation row G. The later graph's 25 names (19+6) contribute once. Endpoint flow G, forward-only dummy gradients and four fixed buffers must be excluded even if physically present in a source tape.

## Proposed next TESTONLY slice, not authorised yet

Propose a **single-fixture, ungrouped computed-boundary join**, with no changes to the accepted coordinator/adapter and no mock G commits. Candidate arrays stay local until all phases and final comparisons succeed. It would establish the missing numerical connection without prematurely claiming a multi-phase owner fence.

1. Verify schema/pin/config/exact names/extents and freeze fixture parameters, fixed buffers, batch, mask, noise and times once. Use `trainingStepModelsFromOracle` only as fixture construction; no `PocketTrainingStep` calls to manufacture boundaries or expected outputs.
2. Use zero-seed public weighting/backbone tapes through the accepted error adapter to compute logvars and audio hidden. Compute unweighted EOS mean with the public EOS projection tape. Forward gradients are ignored, not supplied as expected G. Keep all current source/adapter/payload/metadata limits.
3. For valid rows in fixed order, compute diagonal/distillation forward arrays with public builders; pass those owned values into the backward builders. No upstream fixture gradient, logvar derivative or final input derivative enters a seed or boundary. Decode/validate all row outputs; collect complete named flow candidate gradients and row reduction fields. Inactive fields remain zero. Raw distillation and already-scaled diagonal semantics follow the map above.
4. Build `BuildStepBackboneTrainingTape` and `buildStepResultsTape` using only those computed row fields/logvars/hidden/EOS mean. Evaluate using the accepted adapter, decode real selected result policies, extract the exact later 25-name gradient map and join it with the computed flow totals.
5. Compare all five metrics, full audio/voice/logvar derivatives and exact 59-name G against `training_step_pytorch.json` at the established gates. Report valid forward-logvar prefix errors separately pending approval of that output check. Reject missing/foreign/duplicate/misassigned roots and supplied-fixture row/G injection. Report per-family mean/RMS/max, with no update/checkpoint test.
6. Optionally add a separate one-row minimal-LSD check using its own pinned fixture/weights and existing tighter row gates. It diagnoses mixed AD/stop-gradient; it must never substitute for the full-step result.

The first proposed slice has zero prior G and no accumulation-order/group equivalence claim. Scale 0/.25/1 with prior G and two group partitions is already accepted for independent component fixtures; the full-step oracle only covers p=.75/EOS=.1. New endpoint/objective variants require an independent reference decision rather than reusing this golden result at another config.

## Gates, missing fixtures and structural blockers

| Quantity | Acceptance for proposed computed join | Current independent availability |
|---|---|---|
| Final full-step metrics | Existing absolute `3e-5`; explicit finite checks | Complete-step fixture, five metrics |
| Final logvar/audio/voice derivatives | Existing absolute `5e-5`; finite/extent; exact padded logvar derivative zeros and real tail equality | Complete-step fixture |
| Full-step total named G | Existing absolute `8e-5`; exact 59-name set/606 extents; finite; per-family distributions | Complete-step fixture; do not override tighter separate component gates |
| Valid forward logvars | Report finite-aware error distribution against exported valid outputs; an independent forward-output gate needs explicit calibration/approval. Existing `2e-6` weighting composition checks remain schedule diagnostics | Fixture has two valid outputs per branch; no padded forward-output golden |
| Same-geometry per-row condition/target VJPs, raw stats and per-row flow-G | Ordinary CPU/F64 diagnostic only under existing diagonal (`5e-5`) / distill (`2e-5`, G `3e-5`) / stats (`2e-6+2e-6*abs(ref)`) gates | Full-step JSON does **not** export these intermediates; separate minimal-LSD fixture is different geometry |
| Same-geometry hidden/EOS/logits/endpoint VJP provenance | Computed schedule, final pinned outputs validate aggregate | No full-step intermediate fixture |
| Full-step noise/time adjoints | Optional row diagnostic at existing row gates; excluded from final result contract | Complete-step JSON has no noise/time gradient arrays; separate minimal-LSD row covers them |
| Empty voice, active last-row nonzero tail, repeated token, p=0/.25/1 full-step independent totals | Existing synthetic/component diagnostics remain separate | Missing independently pinned complete-step variants |

No new forward-logvar tolerance is set by this plan. Implementation approval must settle that missing acceptance check without silently widening any established gate. The main full-step gradient oracle is already available, so regenerating upstream assets is unnecessary for the initial final-total proof. Exporting same-step intermediate row arrays would require a separate generator scope and provenance review; none is requested here.

A full stitched tape is not proposed now: per-row early phases plus the later graph can exceed tensor/command/metadata limits, and this read-only work did not construct graphs or measure fit. The previously accepted component flow grouping reserved 953,732 bytes; it leaves no basis for assuming retained full-step plans fit 1 MiB. Preserve current limits and report exact first binding constraint if an approved future fixture cannot fit.

The accepted coordinator covers one source/exact selected-G union. Repeated diagonal/distillation contributions share the same 34 names, and later source constants depend on earlier execution outputs. Calling its `run` independently per phase would not provide one all-phase fence or all-phase admission. A future grouped full-step claim therefore needs a separate design for immutable computed-boundary summaries, repeated-contribution identity/scaling, complete before-first-commit admission, failure-after-progress/publication and host plan/candidate lifetimes. No native resident or optimiser bridge is needed or authorised to solve that test-only problem.

## Read-only verification and return

Static inspection read the existing builders, reducers, decoders, CPU reference paths, test tolerances, JSON metadata/name/extents and upstream fixture generator; the generator was not executed. The corrected local coordinator/alias test still matches the accepted export. No test, build, graph/numerical gate, asset or model/audio work was run. A delegated fixture audit timed out without usable findings and supplies no independent approval. A separate narrow read-only review of this note found no plan blocker in the inventory, scaling/stop-gradient, golden-input separation or proposed scope; implementation still requires `@default`'s next decision.

Request `@default` review of the boundary map and proposed one-fixture ungrouped join before any implementation. Optimiser/checkpoint/production resident/native API/CLI, ceiling/guard/download changes, assets/GPU/released-model/audio/recovery/listening/publication remain held. Overall grouped full-step and native proof remain open.
