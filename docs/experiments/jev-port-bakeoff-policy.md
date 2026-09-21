## Native JEV-equivalent port bake-off

This experiment identifies the best JEV-like choice/reranking model among the native OpenJEV, Decider, Laya and Nimble ports. It is separate from the closed Jevlike Qwen3 experiment and must not use that experiment's 1,440-row final set for model selection.

## Data boundary

Use only the existing `pilot-v4` validation partition. Exclude all 60 originals selected by `direct-pilot-v1`, because their outcomes and controls have already informed development. Freeze the remaining rows into two disjoint cohorts before scoring any candidate:

* screening takes eight originals per task by the lowest SHA-256 of `jev-port-bakeoff-v1:source_id`;
* finalist evaluation uses every remaining untouched validation original;
* both cohorts retain source/config boundaries, original candidates, labels and stable candidate IDs;
* reverse-option controls preserve candidate IDs and text;
* MultiNLI and CLINC also receive bounded shuffled-evidence and no-evidence controls.

The script must verify the dataset manifest, validation partition, provenance and spent-request hashes before writing either cohort. The generated selection records every hash and row count. It may not read `test.jsonl`.

## Common request mapping

Each candidate receives the same state, question, ordered option texts and stable IDs. Adapters use only the released public contract:

* OpenJEV ranks `The correct answer is: <option>` by entailment probability;
* Decider receives one Choice question with stable IDs and option text as descriptions;
* Laya receives one Choice question with the same stable IDs and descriptions;
* Nimble receives one enum field with stable IDs and option text as descriptions.

The state is `Evidence:\n<evidence>` when evidence is present and `No separate evidence supplied.` otherwise. The question text is unchanged. Adapters may apply the released model's existing truncation and temperature rules, but may not tune prompts, temperatures or thresholds against either cohort.

## Two-stage decision

Apply a resource-admission gate before scoring: the released native path must fit within 48 GiB peak RSS and one minute per request on this host. Nimble's already published measurements are about four minutes per field and 59 GiB peak RSS, so it fails before seeing cohort outcomes and is recorded as resource-inadmissible. Score OpenJEV, Decider and Laya on screening. The two highest-accuracy complete candidates advance. Ties resolve by fewer reverse-option answer changes, then lower total decision time.

Score the finalists once on the untouched finalist cohort. Select the winner by:

1. normal accuracy;
2. fewer changed answers under reversed options;
3. for MultiNLI and CLINC, larger normal-versus-shuffled/no-evidence degradation while retaining normal accuracy;
4. lower measured decision time and peak RSS.

Report every candidate, admission rejection and failure. A resource failure is a result, not permission to substitute another checkpoint or alter the cohort. Do not calibrate or train on these cohorts. After model selection, report NLL, summed multiclass Brier score, ten-bin ECE and reliability data from each arm's stored candidate scores normalized within the offered set; these are descriptive held-out metrics, not an additional selection gate or permission to fit a temperature.

## Publication boundary

Publish the frozen selection and adapter source before candidate scoring. Preserve raw logits or probabilities, selected stable IDs, timings, model/checkpoint identities and resource measurements. A later model or prompt change requires another experiment and another untouched cohort. Simple-JEV is excluded until its upstream repository has a compatible licence or explicit permission.
