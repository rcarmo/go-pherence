# Trained Community-1 Vulkan placement — 13 September 2026

The fixed-window hybrid completed the pinned 30-second public sample on Intel Iris Xe. Its segmentation, masks, support data, turns and DER match the retained CPU/fresh-reference result. Strict neural intermediate and tie-policy gates remain open.

## Tested revision and device

- Source revision before this checkpoint: `7d88d82fbdec191ef3653013b091d0faa1f6b244`.
- Device: `Intel(R) Iris(R) Xe Graphics (RPL-P)`.
- ICD: `/usr/share/vulkan/icd.d/intel_icd.x86_64.json`, SHA-256 `d8f5bacf3b35d259cb22fcd70b592d76d1a0b6e0b9003bae855ddfc4d0e917b3`.
- Driver selection used both `VK_DRIVER_FILES` and `VK_ICD_FILENAMES`.
- `GO_PHERENCE_DISABLE_NVIDIA=1`, `GOMAXPROCS=2`, `CGO_ENABLED=0`.
- No matching speech/LLM process or listener was present before the run. The local user-bus environment was unavailable, so `systemctl --user` could not provide unit state.
- Host memory before the run: about 26GiB available. `/usr/bin/time` reports zero process swaps.

The model and public WAV hashes are enforced by the test. PLDA hashes are:

| Asset | SHA-256 |
|---|---|
| `xvec_transform.npz` | `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f` |
| `plda.npz` | `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255` |
| public WAV | `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e` |

Raw PLDA preparation reports condition `56.00115466461734` and two-sided inverse residual `1.7763568394002505e-15`.

## Native run

`TestVulkanCommunity1TrainedDiarization` is disabled by default. It requires an expected physical-device substring, a test deadline no longer than ten minutes and explicit local asset paths. It rejects software Vulkan devices. The test constructs one `NewVulkanDiarization` owner, runs one 480,000-sample input, closes the owner and checks that Vulkan allocation and state counters return exactly to their initial values.

| Measurement | Result |
|---|---:|
| Constructor | 34.691541ms |
| 21-window inference and postprocessing | 27.531066363s |
| Close | 707.823µs |
| Complete test process | 28.05s |
| Maximum RSS | 121,032KiB |
| Tracked live Vulkan memory | 87,219,248 bytes / 2 allocations |
| Post-close Vulkan delta | 0 bytes / 0 allocations |

The retained existing-CPU ABBA median is `49.235s`; the opt-in tiled CPU candidate median is `40.285s`. The native test process is `1.77×` faster than the existing CPU median and `1.45×` faster than the tiled median. These runs occurred in different windows and are not balanced ABBA or energy measurements. The complete process includes hash checks, model loading, inference and result serialization.

## Numerical and quality comparison

The run uses the same explicit diagnostic `LowestIndexTies` policy as the retained CPU result.

| Comparison | Result |
|---|---:|
| Segmentation | 37,107 / 37,107 exact |
| CPU–Vulkan segmentation | bit-identical |
| Fresh-reference embedding max absolute error | `3.3974647521972656e-6` over 16,128 values |
| CPU–Vulkan embedding max absolute error | `9.55e-7` |
| CPU–Vulkan weight/support/mask data | bit-identical |
| CPU–Vulkan postprocess max numeric difference | `3.410396569591967e-7` |
| Training rows / clusters | 37 / 2 |
| Full / exclusive turns | 13 / 12 |
| Ambiguous frames | 84 |

Full DER is `5.2073921971%` at collar 0 and `1.3177976791%` at collar 0.25. Exclusive DER is `10.2909394251%` and `4.9445005045%`. All four values match the fresh reference exactly. Full and exclusive turns match after one global speaker-label mapping with `1e-12s` boundary tolerance.

The scorer writes `qualified=false`: this is one public sample under an explicit diagnostic tie policy, and strict SincNet/embedding intermediate comparisons still fail. The run does not establish broad-corpus DER, default tie behaviour, crash/reopen recovery under trained load, repeated performance, F16/quantised placement, long-file scaling or deployment readiness.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR=/path/to/community1-segmentation-final-3533c8
export GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR=/path/to/community1-embedding-3533c8
export GO_PHERENCE_COMMUNITY1_PLDA_DIR=/path/to/community1-plda-3533c8
export GO_PHERENCE_COMMUNITY1_PUBLIC_WAV=/path/to/pyannote-sample.wav
export GO_PHERENCE_VULKAN_COMMUNITY_DIARIZATION_OUTPUT=/path/to/new-result.json
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-community-trained-check
```

Result SHA-256: `111e98ad0a6d1fb5dc61bcf6db4ff0d3d44485cc66c09efcabf7736b7d8f4ee3`.

Saved scorer SHA-256: `98eb3cab4835616534a76933cc379dfb3eaa2037bdc63bf6a43d8d2bcb618893`.
