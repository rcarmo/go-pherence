# Trained Community-1 Vulkan process recovery — 13 September 2026

A trained native owner survived the required process-level recovery sequence on Intel Iris Xe: the parent killed one child after its first complete Vulkan window, then a fresh child initialized the device, completed all 21 windows and closed without tracked Vulkan state or allocation leakage.

## Scope

`TestVulkanCommunity1TrainedProcessRecovery` is Linux/amd64-only and disabled by default. It requires the same hash-pinned segmentation, embedding, PLDA and public WAV assets as the trained placement gate, an explicit physical-device substring and a test deadline no longer than two minutes.

The first child:

1. initializes the Intel Vulkan ICD;
2. loads and validates the trained assets;
3. constructs `NewVulkanDiarization`;
4. completes one 160,000-sample native window;
5. emits `TRAINED_VULKAN_WINDOW_COMPLETE` before reading the second window;
6. is killed by the parent with `SIGKILL`, so no Go defer or Vulkan close runs.

The second child starts from a new process. It initializes the same physical device, constructs a new owner, completes the full 480,000-sample input, validates 21 windows, 37 training rows, two clusters, 13 full turns, 12 exclusive turns and 84 ambiguous frames, then checks exact return to its initial Vulkan allocation/state snapshot.

This verifies process teardown plus fresh-process device/model reopening. It does not implement in-process device-loss recovery, retain a partial neural-window checkpoint, or prove durable job-store replay with trained tensors. The existing model-free speech-job tests cover quarantine/store ownership and whole-stage retry separately.

## Result

- Device: `Intel(R) Iris(R) Xe Graphics (RPL-P)`.
- Parent test: `29.39s`; wall clock from `/usr/bin/time`: `29.67s`.
- Maximum RSS: `157,688KiB`.
- Process swaps: zero.
- Host `pswpin`/`pswpout`: unchanged at `41812` / `1080673` across the test.
- Child sequence: `killed_after_first_window=true fresh_process_complete=true`.
- Services/listeners: no matching speech, LLM or speechjob process/listener before execution.

The test log SHA-256 is `c3b836690a2d594797d0ee70d6ea64ab4db0ef5af5a0539e8221cdd073de59a5`. The `/usr/bin/time` output SHA-256 is `18b3cb5fdcc871d55edbe89684f7da85382399069b2a7ae16a1970ab7b0db162`.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR=/path/to/community1-segmentation-final-3533c8
export GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR=/path/to/community1-embedding-3533c8
export GO_PHERENCE_COMMUNITY1_PLDA_DIR=/path/to/community1-plda-3533c8
export GO_PHERENCE_COMMUNITY1_PUBLIC_WAV=/path/to/pyannote-sample.wav
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-community-trained-recovery-check
```
