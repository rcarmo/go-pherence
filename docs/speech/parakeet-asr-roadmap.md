# NVIDIA Parakeet ASR roadmap

`nvidia/parakeet-tdt-0.6b-v3` is a candidate for a separate native transcription backend. It is not implemented in go-pherence; no Parakeet weights have been downloaded or executed for this roadmap.

## Pinned candidate

- [Model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/README.md) at revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; the inspected card's SHA-256 is `7a4306a43f395e3a716e77f1d81c03db18a0a355d7b566bb1d6307213ba0302b`.
- NVIDIA describes a 600M-parameter FastConformer/TDT transcription model for 25 European languages, with automatic language selection, punctuation and timestamps. The card expects mono 16 kHz audio. Hub licence metadata says `cc-by-4.0`; review attribution and redistribution requirements before using artifacts.
- V2, CTC, RNN-T, streaming and multitalker Parakeet variants have different contracts. This roadmap selects **v3 TDT** for initial inspection only; it does not treat another variant's model output as an oracle.

## Gates

1. Inspect pinned model and processor manifests without fetching weight payloads. Record all filenames, sizes, hashes, frontend/framing and TDT decoding conventions, including timestamp units, blank/duration vocabulary and language behaviour.
2. Obtain explicit approval before downloading or running the checkpoint. Freeze an independent NeMo reference with exact audio, transcript, token/duration and timestamp fixtures for silence, short/multilingual speech, chunk seams and long input. Keep reference execution separate from native Go inference.
3. Build scalar Go frontend, encoder and TDT decoder slices with bounded memory, deterministic operator fixtures and admission checks. Move measured hot paths to checked SIMD backends after reference parity.
4. Expose it as an explicit ASR provider in the [speech workflow](speech-integration.md), retaining Whisper/MOSS choices. Qualify cancellation, retries, language selection, timestamp mapping, accuracy and end-to-end throughput on labelled recordings and target hardware.

Model metadata and a planning document do not establish native transcription, released-model parity or production speech quality.
