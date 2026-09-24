# Nemotron 3 Diarization roadmap

NVIDIA's `nvidia/Nemotron-3-Diarization` is a candidate for a separate native
speaker-diarisation backend. No checkpoint has been downloaded or run for this
roadmap, and go-pherence has no Nemotron inference implementation.

## Pinned source and scope

- Model: [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/a435e9867d79e789e90053f9b6d6834053af564a), revision `a435e9867d79e789e90053f9b6d6834053af564a`.
- Published [configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/config.json): `Nemotron3DiarizationForAudioFrameClassification`, 31 audio-transformer layers, width 512, 128 mel bins, factor-eight subsampling, and an eight-speaker head. The configuration also defines chunk/FIFO and speaker-cache state; their numerical and temporal behaviour has not been verified here.
- The [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/README.md) describes offline and streaming diarisation for up to eight speakers. The Hub records `openmdw-1.1` licence metadata. Review the actual terms and checkpoint redistribution conditions before using weights or publishing derived artifacts.

The existing [Whisper and Community-1 workflow](speech-integration.md) keeps
plain transcript output independent of optional speaker attribution. Nemotron
would be a distinct opt-in speaker provider, not a silent replacement for
Community-1. Neither backend has trained-quality acceptance in the integrated
speech job.

## Proposed gates

1. Freeze the pinned configuration, processor contract, asset inventory,
   licences and file hashes without fetching model weights. Check sample rate,
   feature framing, channel order, timestamp origin and maximum-speaker rules.
2. Obtain explicit approval before downloading/executing the checkpoint.
   Produce independently generated small fixtures for the frontend, subsampling,
   transformer/head, arrival-order speaker cache, streaming state and offline
   frame-to-turn decoding. Record the oracle software version, input hashes,
   output hashes, units and numerical tolerances.
3. Implement a scalar Go reference with bounded allocations and checked state
   transitions. Promote measured hot operations to checked SIMD backends only
   after operator and whole-recording parity. A Python/NeMo process is an oracle
   option, not the Go inference runtime.
4. Add an explicit speech-job provider selection while preserving plain
   transcript/VTT on diarisation failure. Test overlapping speakers, silence,
   unknown speakers, chunk seams, cancellation/retry, long recordings and
   deterministic turn labelling without reusing Community-1 checkpoints.
5. Measure diarisation error rate on an independently labelled cohort, both
   offline and streaming latency/throughput on each proposed native platform.
   Record speaker-count and recording-length limits. Do not infer accuracy from
   model metadata or synthetic operator tests.

This is a planning item. There is no native execution, released-model parity,
trained-quality result, production latency measurement or readiness promotion.
