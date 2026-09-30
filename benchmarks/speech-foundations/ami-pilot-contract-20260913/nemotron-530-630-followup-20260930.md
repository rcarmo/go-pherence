# Nemotron AMI ES2004a non-overlapping 100-second follow-up — 30 September 2026

The first six task/backend runs on the AMI 530–630-second headset-mix excerpt each finished within 100 seconds on the tested i7-12700/RTX 3060. The **repeat Vulkan diarization request failed**: `vkWaitForFences: -4` followed by an NVIDIA Xid 79 GPU-fell-off-bus event. The device was unavailable afterwards. Stop PTX/Vulkan/CUDA tests on this host until the operator has restored and checked it. No GPU recovery or further GPU test is part of this report.

This excerpt does not overlap the [350–450-second follow-up](nemotron-100s-followup-20260930.md), but both come from meeting `ES2004a` with the same mix, speakers, acoustic conditions and annotations. Neither is a corpus-level quality or reliability estimate. The [AMI source contract](README.md) pins the official CC BY 4.0 audio and annotation archive. The local `scripts/prepare_ami_corpus.py --meeting ES2004a --start 530 --duration 100` output matched the committed [asset contract](nemotron-530-630-result.json): 1,600,000 samples, 36 RTTM turns and 269 annotated word objects. The canonical WAV SHA-256 is `4fbea62ac20148203c75ab2e094dcb902beb67f5504d566430c8cc8702d3a69e`; RTTM `ba3aac01bb0fa6bc84d4d58314c04cbe7a9fbde57f24568822c54a514aacbb34`; words `2430219f312e05c5a3137c2c056a22c0ceb717cd7be638101584d412cb6cb02e`. Media, RTTM and words stay outside Git.

A binary built from clean `ede9ea019fad6732962c9dddc49578fc1add9c90` with `go build -o /workspace/tmp/nemotron-ami-es2004a-350-410/nemotron-cli ./cmd/audio/nemotron` ran with `GOMAXPROCS=4`. The same binary, model/checkpoint pins and loop/flags used in the [350–450-second report](nemotron-100s-followup-20260930.md) applied here, with `p=/workspace/tmp/nemotron-ami-es2004a-530-630` and `-input "$p/ES2004a-530-630.wav"`. `/usr/bin/time -f 'wall_elapsed=%e maxrss_kb=%M'` measured the entire process, including WAV and checkpoint load but excluding compilation.

| Backend | ASR wall / peak RSS | Diarization wall / peak RSS |
|---|---:|---:|
| SIMD | 59.28 s / 5,178,184 KiB | 84.85 s / 1,588,096 KiB |
| PTX | 60.37 s / 5,294,344 KiB | 24.34 s / 1,718,688 KiB |
| Vulkan | 61.20 s / 5,351,200 KiB | 69.10 s / 2,738,968 KiB |

The successful first Vulkan diarization run was **69.10 seconds**, versus 30.93 seconds on the earlier 350–450-second excerpt, although both contain 100 seconds of PCM. Shape/input-dependent work or host/device conditions require diagnosis; a single pair of runs cannot assign cause. ASR PTX/Vulkan are CPU/GPU projection hybrids, not fully GPU ASR. The opt-in resident PTX/Vulkan diarization towers keep frontend, audio layer 0, head and speaker cache on CPU.

After the GPU failure, three **CPU-only** repeats of SIMD diarization used the same CLI binary/input/model and `GOMAXPROCS=4`. Their process wall times were **74.70, 75.08 and 75.56 seconds** (initial 84.85 seconds); each exited successfully and produced the same parsed 67-span geometry as the initial run. The four-run range is 74.70–84.85 seconds, still below the 100-second audio duration. The first-run difference was not isolated. A separate CPU-only profile of the existing 30-second JFK PCM stream benchmark took 11.51 seconds per request with 6,003,984,080 B/op and 82,021 allocs/op; `gebpMicroKernel` had 64.40% flat sampled CPU, and the prepacked SGEMM path 69.87% cumulative. This profile is a cost lead, not a speedup or explanation for the initial 84.85-second run. Local evidence: `$p/simd-diarization-{2,3,4}.{time,log,json}` and `$p/diar-{cpu.pprof,profile-bench.txt}`. No GPU workload was run for these checks.

All three *successful* ASR runs produced 200 whitespace tokens and all three *successful* diarization runs produced 67 spans. Saved transcript SHA-256 `7406c54a682aac4620f5078543fc19792127d9020ce490fca29c647efc109622`; saved spans `79be183ed87ac997fb40b91d4fa1f9e2d1b96e2d140c1a0de12213c1be5dd489`. These identify local output files for replay, not numerical acceptance gates. The later failed Vulkan run produced no spans and is excluded from the scores.

The [checked score](nemotron-530-630-score.json) uses `scripts/score_nemotron_ami_pilot.py`, the [committed asset contract](nemotron-530-630-result.json), independent reference words and RTTM, NFKC/casefold tokenisation, a `[0,100]`-second UEM and `pyannote.metrics==4.1` with overlap included. The 269 annotated word objects normalise to **274 reference tokens**; 200 hypothesis tokens give transcript-only WER **88/274 = 0.321168** (12 substitutions, 75 deletions, 1 insertion). DER is **0.392649** at zero collar (45.178 missed and 0.008 false-alarm speaker-seconds, zero confusion against 115.080 reference speaker-seconds) and **0.359423** at 0.25-second collar. JER is **0.394333** and **0.365048** respectively. These are absolute excerpt scores; there are no hypothesis word times, speaker-attributed WER or pinned independent reference-system deltas. Backend agreement on saved outputs does not establish labelled quality.

## GPU failure and recovery gate

The second Vulkan diarization invocation used the same binary, WAV and checkpoint:

```sh
p=/workspace/tmp/nemotron-ami-es2004a-530-630
GOMAXPROCS=4 /usr/bin/time -f 'wall_elapsed=%e maxrss_kb=%M' \
  -o "$p/vulkan-diarization-2.time" \
  /workspace/tmp/nemotron-ami-es2004a-350-410/nemotron-cli \
  -task diarization -backend vulkan -vulkan-tower \
  -input "$p/ES2004a-530-630.wav" \
  -model checkpoints/nemotron/diarization/model.safetensors \
  > "$p/vulkan-diarization-2.json" 2> "$p/vulkan-diarization-2.log"
```

It exited nonzero after 24.09 seconds with no JSON output. The stderr guard reported `Nemotron Vulkan tower layer 29: Vulkan submission still in flight`, `Vulkan device lost; process-level recovery required` and `vkWaitForFences: -4`; cleanup also reported device loss. At **2026-09-30 03:16:22 WEST**, the kernel log reported `NVRM: Xid (PCI:0000:00:10): 79, GPU has fallen off the bus` and then Xid 154 (`PF FLR` recovery action). `nvidia-smi -L` subsequently failed to determine the GPU handle. A bounded copy of stderr, timing, kernel events and post-failure `nvidia-smi` output is [committed here](nemotron-530-630-vulkan-failure.txt). Full local failure artifacts are `$p/vulkan-diarization-2.{log,time,json}`; the JSON file is empty. The owner of host recovery was notified immediately. GPU code, the driver, or host conditions cannot be isolated as the cause from this one failure. Preserve these logs and avoid further GPU workloads, driver reloads, device resets or reboots without operator coordination. Earlier completed results remain historical evidence, but the device is not currently available for their repetition or broader GPU qualification.
