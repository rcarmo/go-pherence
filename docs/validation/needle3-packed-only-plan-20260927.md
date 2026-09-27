# Needle 3 packed-only planning check

`planPackedOnly` inventories an already decoded deployed model. It returns sorted tensor names, logical decoded bytes and reasons for two groups: whole tensors with packed execution-site coverage, and tensors still requiring decoded F32 storage. It does not change the model, archive loader or inference path.

The tiny `needle3.cact` fixture has 69 decoded tensor names and 30 retained CQ matrices. The earlier [archive coverage audit](needle3-packed-only-coverage-20260927.md) counted 15,104 of 25,012 decoded bytes with a packed counterpart. The planning check classifies **14,592 bytes as potentially replaceable** and **10,420 bytes as decoded-required**. The 512-byte difference consists of `embedding_head/proj/kernel` and `confidence_head/proj/kernel`: archive parsing retains packed matrices for them, but `probeHead` calls `headWeight`/`param` and consumes their decoded values. Counts describe only the tiny fixture's logical tensor data. The 156,864 retained packed bytes are additional; no RSS, peak-load or production-size saving was measured.

The check recognises `embedding/embedding`, attention projection kernels, MHC phi matrices and engram key/value projection kernels. A layered tensor qualifies only when every layer has a packed matrix of matching execution geometry; partial coverage keeps the entire decoded tensor. Auxiliary-head packed projections have their archive geometry checked but remain decoded-required. Unrecognised packed keys, invalid layer indices, missing expected tensors and mismatched shapes fail the check. Sorted output and a checkpoint before/after comparison test deterministic, non-mutating inspection. A separate test removes one packed layer and checks the explicit partial-coverage reason. The fixture check uses the existing hybrid archive; it does not construct a packed-only checkpoint.

Decoded storage is still necessary at these boundaries:

- `LoadArchive` materialises and validates F32 records before creating the hybrid model. `newModel` requires data for all expected shapes.
- `execution.trunk` gathers token embeddings from decoded `embedding/embedding`; `CQMatrix.DecodeRow` exists but is not wired into the trunk.
- `execution.param` and `NewDecoder` prepare decoded views for every expected parameter, even when `e.linear` or the output projection later selects CQ. Auxiliary-head `probeHead` uses decoded weights.
- `SliceDepth` copies decoded tensors and remaps packed keys; `Checkpoint()` exports owned decoded tensors. Neither has a packed-only policy.

A future packed-only representation needs independent shapes and strict consumer routing, a policy for heads, depth slicing and export, and bounded archive validation/loading without keeping both expanded and packed data. It then needs independent full/cached/head parity, concurrent ownership and cancellation tests, actual peak/retained memory, production-size CQ2/CQ4 performance, and native amd64/ARM64/RVV qualification. The decoded archive and execution path remain the default.
