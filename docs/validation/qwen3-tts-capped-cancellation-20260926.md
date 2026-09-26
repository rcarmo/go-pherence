# Qwen3-TTS capped CPU cancellation

`GenerateCappedSeededCPUContext` adds an opt-in context-aware path for the
bounded 0.6B CustomVoice CPU runtime. The existing seeded, greedy and fixed-frame
entry points keep their previous signatures and behaviour. Cancellation returns
`context.Canceled` or `context.DeadlineExceeded` with an empty result: no partial
semantic codes, acoustic codes or waveform escape.

The implementation checks the context before model work, after Talker prefill,
after the first acoustic frame, between Talker layers, at each continuation
frame, after sampling and acoustic prediction, before decoding, and before
returning decoded output. An individual Talker layer, CodePredictor frame and
waveform decoder remain non-interruptible. Cancellation can wait for one such
operation to finish; this API makes **no hard latency guarantee**. A request
cancelled during decoding may consume decoder CPU before returning an empty
result. The new path does not start background workers or mutate shared model
weights.

The synthetic two-frame suite cancels before work, checks an expired deadline,
and deterministically cancels at the first and second token-selection
boundaries. It then generates a successful request on the same model and
compares every result field with the unchanged legacy seeded API. Eight
simultaneous context-aware requests return independent owned outputs. The
focused test passed `-race -count=10`.

With the approved, hash-checked CustomVoice checkpoint, a pre-cancelled and
expired two-frame “Hi” request returned an empty result. Successful
`Hi`/Ryan/English/seed-42 context and legacy requests both returned semantic
`[1995, 215]`, 30 acoustic codes and 3,840 samples, with all fields equal. The
released ordinary test passed in 9.18 s; the released race test passed in
37.02 s. This is a short released recovery check, not a measured cancellation
latency or a 64-frame cancellation test.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts \
  -run '^TestCappedSeededCPUContextReleasedHiTwoFrames$' -count=1 -v
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go test -race ./model/qwen3tts \
  -run '^TestCappedSeededCPUContextCancellationAndRecovery$' -count=10
```

Evidence is under `/workspace/tmp/qwen3tts-cap64-20260926/`:
`context-synthetic-race10-final.log`, `context-released-short.log` and
`context-released-short-race.log`. No GPU execution ran. Cancellation inside
codec kernels, deadline-to-return distribution, multiple long real requests,
streaming and hours-long retention remain unqualified.
