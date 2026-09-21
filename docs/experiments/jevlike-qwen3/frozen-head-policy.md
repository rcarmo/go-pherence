## Frozen-head learning check

The direct study's residual order failures and 1.5--3.2-second prefills justify a
small late-interaction comparison, not an automatic expansion to all prepared
training data. Keep the installed instruction checkpoint, the 12 GiB combined
pilot-cache cap and the 30 GiB free-disk floor.

Select 32 training examples per source/config using the existing
`direct-pilot-v1:source_id` hash order within **train only**: 160 examples in
all. Use the same 60 validation originals and 228 control requests as the direct
study, and the existing separate 60 calibration originals. No final-test reads.
The balanced selection makes epoch sampling balanced without replacement.

Contexts are `Evidence:\n{evidence}\nQuestion:\n{question}`. Each candidate text
is encoded separately. Both use plain checkpoint tokenisation, no BOS/EOS and no
chat wrapper; keep every final-normalised causal context row and mean-pool option
rows in F32 before storage. Reject context lengths above 512 or options above
128; never silently truncate. The direct and head inputs contain identical
question/evidence/candidate text but deliberately differ in architecture and
formatting. This is a same-backbone comparison, not prompt-equivalent execution.

The initial cache uses F32. A separate FP16 conversion keeps pooled options and
changes the contract identity; compare feature and head-logit differences before
using it. Cache keys bind exact text, token IDs in checksummed metadata, verified
checkpoint file identity, dataset identity, representation/backend, pooling,
dtype and token limits. Stored features are unprojected -- no trainable keys,
values or layer-norm outputs are reused between optimiser updates.

Use native CPU head forward/backward, rank 64, batch 4, AdamW learning rate 0.002,
clip norm 1 and five epochs. Fixed seeds are 7, 17 and 27. Select each seed's best
validation-NLL epoch, reporting all seeds rather than selecting one by accuracy.
No hyperparameter sweep. Save current weights, Adam moments, step, epoch, batch
cursor and partial loss after every update; best weights are separate. Epoch
shuffle is regenerated from seed+epoch and a recorded algorithm identifier.
An intentional one-step interruption must match uninterrupted training exactly.

Expansion requires mean validation accuracy at least random plus ten percentage
points, and at least ten percentage points normal-over-shuffled evidence gain on
**both** NLI and CLINC. These are exploratory gates with only 12 validation
examples per task; report every failure and do not lower the requirement after
seeing results. A losing head is a completed learning check, not a reason to hide
runs or increase the dataset automatically.

Published benchmark data cannot substitute for independently human-reviewed new
examples. Machine-authored probes may test fresh-input mechanics but must not be
labelled human-reviewed. That acceptance item requires actual human review.

[Direct results](direct-study.md) | [Experiment status](README.md)
