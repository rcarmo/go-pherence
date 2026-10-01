# Native Silero VAD

This package implements the native CPU graph, fresh recurrent state, speech-span policy and original-sample mapping for the retained Silero 6.2.0 Whisper VAD checkpoint. It is experimental and available through the checked [Whisper VAD API](../whisper/pcm_vad.go). It is not a serving default. Nine bounded independent whisper.cpp fixtures passed probability, segmentation, reset and cancelled-retry checks. Explicit gap-preserving Whisper VAD completes bounded multilingual/word-timing fixtures. Independent recognition/timing accuracy and speed acceptance are incomplete.

The [legacy GGML loader](../../loader/silero/ggml.go) requires a pinned hash before decoding. It admits the 15-tensor inventory and scalar final bias, owns decoded F32 values, and rejects malformed geometry, unsupported dtypes, nonfinite weights and trailing data. `New` copies immutable weights; each `NewStream` owns scratch and LSTM state.

## Numerical contract

The implementation follows the graph in whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`: one 512-sample frame, reflect-64 STFT padding, four convolution/ReLU layers, LSTM, ReLU and final sigmoid projection. That graph reads the checkpoint's 64-sample context metadata but does not prepend context to its input. It differs from the PyTorch 576-sample schedule; parity with one must not be inferred from the other.

F16 convolution weights and their im2col activations retain the source's binary16 rounding boundary. CPU reductions use `simd.Sdot32`, which preserves the pinned original's four eight-lane FMA accumulators, tree reduction and F64 scalar-tail addition. The existing generic `Sdot` has a different reduction order. Unsupported SIMD platforms use the explicit scalar FMA tree. Running these reductions is genuine Go/backend inference; no external engine supplies probabilities.

## API and state

- `Probability(ctx, frame)` accepts exactly 512 finite mono 16-kHz samples. Errors/cancellation leave recurrent state unchanged. Fixed stream-owned scratch requires no steady-state heap allocation in the synthetic test.
- `Reset` clears recurrent state; `NewStream` creates independent state sharing immutable weights. Streams must not be used concurrently.
- `Probabilities` starts fresh state, reads bounded 64K-sample batches and zero-pads only the final tail. It returns no partial scores on failure.
- `SpeechSpans` applies the selected threshold/hysteresis, minimum speech/silence, short-gap merge and padding policy. Output is clipped to actual PCM samples. Finite maximum-speech splitting is not implemented.
- `CompactedReader` reads retained spans without holding the entire PCM. `MapStart` and `MapEnd` preserve different boundary directions between speech spans. A word/cue crossing a removed gap needs explicit caller handling; this package does not invent alignment.

The existing no-VAD/resume Whisper APIs still only skip exact zero PCM. The opt-in API uses fresh native VAD state and returns retained-window coordinates separately from original segments, words and speech spans. Compact mode rejects words crossing removed silence. Explicit `PreserveWindowGaps` retains actual internal silence in each original-audio group for recognition and alignment; disjoint groups decode separately. It exposes no resume API and changes no checkpoint or serving default. See [bounded gap-preserving qualification](../../docs/validation/vulkan-attention-key32-20261001.md). Toy integration tests establish orchestration and mapping only; they do not establish trained recognition quality or performance acceptance.

## Verification

Ordinary tests are small, synthetic and offline. They cover loader errors/pinning/ownership, scalar F16/F32 constants, recurrent state, reflective padding, convolution layout, cancellation rollback, speech/silence/hysteresis, malformed options, tail padding and gap-aware PCM/timestamp mapping. Test timing is not a performance gate.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./loader/silero ./model/silero -count=10
```

The optional `TestPinnedSileroFile` reads and decodes the pinned model without neural execution. Set `GO_PHERENCE_SILERO_MODEL` to opt in.

The [independent reference helper](../../scripts/silero-vad-reference.cpp) uses the original public VAD API and is only an offline qualification oracle. It must be built against the pinned original header/library in an authorised CPU window. It is never invoked by Go inference. Input is bounded mono16-kHz little-endian raw F32; output contains probabilities and centisecond speech spans.

`TestPinnedSileroReferenceProbabilities` requires:

- `GO_PHERENCE_SILERO_REFERENCE`: oracle JSON.
- `GO_PHERENCE_SILERO_MODEL`: pinned GGML asset.
- `GO_PHERENCE_SILERO_PCM`: matching raw F32 input.
- `GO_PHERENCE_SILERO_PCM_SHA256`: input pin.
- `GO_PHERENCE_SILERO_TOLERANCE`: explicitly audited JSON with `MaxAbsolute` and `MeanAbsolute`.

The reference gate checks score errors, threshold decisions and exported boundaries. No tolerance is chosen or widened automatically. The gate passed ten repetitions each on JFK, complete silence, attenuated JFK, JFK with 15 seconds of silence at each end, a short island, synthetic self-overlap, two Portuguese fixtures and one French fixture. All start/end hysteresis decisions matched. Export comparison uses the original's nearest-centisecond rounding and clips its padded EOF to actual PCM. Exact sample spans from the two sets of probabilities also match. See [bounded qualification evidence](../../docs/validation/native-silero-parity-20261001.md). These fixtures do not qualify natural overlap, long-form transcription or the checked Whisper integration's word timestamps.

Goal and matched-engine gates: [native Go Whisper contract](../../docs/speech/whisper-go-performance-contract-20260930.md).
