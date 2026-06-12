# NPU Whisper large-v3 — investigation notes

Goal: run Whisper **large-v3** faster than realtime (RTF < 1) on the MilkV Jupiter 2
(SpaceMIT K3, 8× X60 RISC-V, RVV + IME int8 + NPU `/dev/ai_dma`). Optimize the
runtime, not swap the model.

## Files

| File | Purpose |
|---|---|
| `whisper_npu.cpp` | C++ ONNX Runtime runner. SpaceMIT EP via `SessionOptionsSpaceMITEnvInit`. Modes: full (`enc dec mel`), `--enc` (encoder→H.bin), `--dec` (H.bin→tokens). KV-cached decode loop. |
| `whisper_cpu.py` | Python reference pipeline: mel → encoder → KV-cached merged-decoder decode → detok. Validates transcript. |
| `mel.py` | log-mel-128 features `[1,128,3000]` (matches HF WhisperFeatureExtractor). |
| `detok.py` | GPT-2 byte-level BPE detokenizer from `tokenizer.json`. |
| `turbo_decode.py` / `enc_h.py` | large-v3-**turbo** decode test + turbo-encoder H generator. |
| `quant_static.py` | static int8 QDQ attempt (calibrate on real mels). |

## What works

- **Full ONNX pipeline validated** (CPU): transcript byte-identical to the Go large-v3
  output on a 30s dense-speech window.
- **Merged-decoder KV cache protocol** (Xenova/onnx-community export quirk):
  - Prefill: `use_cache_branch=False`, all past 0-length, request `logits` + **all**
    present (incl. `present.*.encoder`). Encoder KV is produced here.
  - Cached steps: `use_cache_branch=True`, feed back decoder KV (growing) + encoder KV
    (fixed). **Do NOT request `present.*.encoder`** on cached steps — its `Reshape_4`
    in this export throws a stray-0-dim error. Encoder KV is unchanged after prefill.
  - Result: decode 285s → **37.9s** (7.5×) on CPU for 125 tokens.
- **C++ runner correct on CPU/CPU** (125 tokens match reference).
- **NPU int8 matmul ≈ 8.5× CPU** (640 vs 143 GMAC/s microbench). `enc_s8` int8 encoder
  on NPU ≈ **14.2s/window (encoder RTF 0.47)** even with attention on CPU.

## NPU blockers found

1. **21GB OOM:** the encoder's `encoder_attentions.*` graph **outputs** (32×[20,1500,1500]
   F32 ≈ 5.7GB+) are materialized by the EP (it doesn't prune unrequested outputs like
   the CPU provider does). Fix: strip them with `onnx.utils.Extractor` → 2.4GB RSS.
2. **TCM overflow:** the *clean* (no-attention-output) encoder fuses a bigger NPU
   subgraph that exceeds per-core TCM → `tcm buffer acquire failed for core id N`.
   The attention-on-CPU `enc_s8` partitioning fits TCM but needs 18GB.
3. **Static quant OOM:** MinMax calibration instruments every activation (incl. the big
   attention tensors) → augmented model materialization hits 31GB. Dead end on this RAM.
4. **TCM leak / wedge:** aborted runs (uncaught EP `std::runtime_error`) leak TCM; the
   driver does not reclaim it on process death → subsequent runs fail `tcm buffer
   acquire` until **reboot**. Always release the NPU cleanly (guard exceptions).

   EP env knobs: `SPACEMIT_EP_INTRA_THREAD_NUM`, `SPACEMIT_EP_DISABLE_OP_{TYPE,NAME}_FILTER`,
   `SPACEMIT_EP_DUMP_SUBGRAPHS`, `SPACEMIT_EP_ENABLE_DMA`, `SPACEMIT_EP_DENSE_ACCURACY_LEVEL`.

## large-v3-turbo findings

- Turbo's **encoder is NOT identical** to large-v3's: feeding large-v3 H into the turbo
  decoder produces repetition garbage; turbo's own encoder H decodes correctly.
- Decode (ONNX, 6 threads, 30s window, 125 tokens): large-v3 32L uint8 **37.9s**;
  turbo 4L int8 **17.8s (2.1×)**; turbo 4L fp16 **26.7s** (fp16 casting slow on this CPU).
- 8× fewer decoder layers → only ~2.1× faster decode (per-step KV/dispatch overhead).
- **Turbo reuses the same 32-layer encoder** → encoder is still the wall
  (95s ONNX CPU / ~58s Go / 14.2s NPU). Encoder-only RTF on CPU ≈ 1.9, so
  **RTF<1 is impossible on CPU regardless of turbo** — the NPU encoder is mandatory.

## Path to RTF<1

NPU `enc_s8` encoder (14.2s, on a clean/rebooted NPU) + fast turbo int8 decode (~5s with
lower per-step overhead) ≈ **RTF 0.6**. Requires: clean NPU TCM (reboot + exception-safe
release) and trimming decode per-step overhead.
