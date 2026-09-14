# Experimental Community-1 CPU owner

`NewOwnedCommunity1Stage` adds exclusive lifecycle ownership around the existing pure-Go `ExperimentalDiarization` and whole-result durable stage. The raw `NewCommunity1Stage` remains available and unchanged. This owner changes no model modes, tolerances, tie policy, output schema or stage version.

`RunPCM` is synchronous Go with call-local scratch. Cancellation/error/panic unwinds the call before the owner gate is released; no native submission or background worker survives return. Calls serialize. `Close(ctx)` stops new admission, waits for an active call to return, then clears the composition's segmentation, embedding and PLDA references. Timeout or cancelled close leaves admission open unless close had already validly begun. Repeated close is idempotent.

Inference panic poisons future admission because model consistency is unknown, but close can safely clear retained Go references after stack unwinding. Release panic is marked poisoned and close remains retryable. Deterministic reclamation requires exclusive transfer of all model aliases; clearing this composition cannot clear caller-held aliases. Cleared slices/graphs become GC-eligible; no immediate RSS reduction is promised.

The existing stage hashes complete configuration, including output/resource cap, because the cap changes whether the stage can succeed. PLDA SHA256 remains mandatory even though silence/single-training-row inputs may not execute PLDA: the profile attests the PLDA intended for multirow input and does not infer a nil sentinel.

Tests cover serial calls, waiting cancellation, active-close timeout, inference/release panic, retry/idempotence, model graph validation/release and unchanged raw/owned stage identity. They are model-free. Strict SincNet/embedding/tie and trained DER gates remain unchanged and unresolved. No Community-1 server profile, GPU graph or long-file support is added by this slice.
