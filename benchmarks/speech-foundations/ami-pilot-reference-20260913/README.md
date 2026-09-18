# Pinned Community source oracle on AMI — 13 September 2026

## Result

The pinned PyAnnote Community-1 source pipeline was run on the same hash-pinned 60-second `ES2004a` AMI excerpt as the Go implementation. It used the already-provisioned local PyAnnote 4.0.7 / PyTorch 2.14.0+cpu environment, one Torch thread, MKLDNN disabled, batch size 1, weights-only raw checkpoint loading and six exact source-file hashes. No network, GPU, service or untrusted pickle loading was used.

The source pipeline also chooses **2 speakers** for the four-speaker AMI reference and emits the same 22 full/16 exclusive turns as Go. Absolute full-turn DER/JER are 52.7955%/70.9645% at collar 0 and 49.9681%/70.3948% at 0.25 s. The poor automatic-count quality is therefore shared by the pinned source pipeline on this headset-mix excerpt.

## Go/source parity

- Segmentation: **90,117 / 90,117 values exactly equal**, max absolute delta 0.
- Embeddings: 8,746 / 39,168 values exactly equal; max absolute delta **4.0531158×10⁻⁶**.
- Full turns: 22 vs 22, exact boundaries and labels under one permutation.
- Exclusive turns: 16 vs 16, exact boundaries and labels under the same permutation.
- DER delta: exactly 0 percentage points at collars 0 and 0.25, full and exclusive.
- JER delta: exactly 0 percentage points at collars 0 and 0.25, full and exclusive.

This is strong source-parity evidence for one real meeting excerpt. It does not erase the separate strict intermediate failures on retained fixtures, and it does not qualify absolute model quality or the default tie policy.

## Runtime and resources

- Source pipeline wall: **133.58 s**.
- Maximum RSS: **775,448 KiB**.
- Process swaps: 0.
- Host package RAPL delta: **2,010.799 J**, approximately **15.045 W** mean over the bounded command.
- Raw `psys` delta: 28.159 J.
- Endpoint package temperature: 46→51 °C.

The Go Community-only diagnostic took 106.51 s and 160,668 KiB peak RSS on the same excerpt, but this is not a balanced performance comparison: the source run includes Python/PyTorch startup and artifact-hook writes, and the Go run serializes a large diagnostic result. Only output parity is asserted here.

## Decision

The AMI pilot separates two issues:

1. **Implementation fidelity:** Go matches the pinned source result on this excerpt within the existing embedding tolerance and exactly at segmentation/turn/metric endpoints.
2. **Model/policy quality:** both implementations collapse four speakers to two, score poorly, and the Go strict reconstruction policy rejects ties rather than publishing them.

The complete plan remains unqualified until broad corpus quality and default strict behavior pass or the acceptance policy is explicitly revised with independent evidence.
