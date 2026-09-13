# go-264 AC-3 5.1 provider repin — 13 September 2026

The consumer now pins `github.com/rcarmo/go-264@v0.0.0-20260913215118-afd03f5c69ed` (`afd03f5c69edd1410fa4324786f8bc6a95fe39cf`), module sum `h1:pyW5AHeXClSDg7tw3Jd4TMLC4yGmW7tpQsMD7RtGpck=`.

Public provider CI run <https://github.com/rcarmo/go-264/actions/runs/34785055825> passes amd64 default/purego/vet/full-race/FFmpeg-oracle coverage and native ARM64 default/purego/vet/race coverage.

The release adds qualified progressive MP4/MOV AC-3 (`ac-3`/`dac3`) mono through 5.1 support. The retained BBB source contains **19,817 MP4 AC-3 packets/syncframes** and decodes to **30,438,912 PCM frames**. Five retained output policies cover mono, stereo, centre, centre+LFE and surrounds. E-AC-3, fragmented/protected media and unsupported QuickTime entries fail closed.

The go-pherence adapter continues to pass default `audio.Options`: nil `TrackIndex` preserves AAC-first selection and nil `SourceChannels` preserves deterministic mono/stereo downmix. This repin therefore adds provider capability without silently changing server profile semantics or exposing exact track/channel selection. A future public profile option would require its own checked identity, bounds and UX contract.

With CGo/NVIDIA disabled and `GOMAXPROCS=2`, all focused `loader/audio/media`, speech-job, HTTP, CLI and server package tests pass; affected vet and `go mod verify` pass. No FFmpeg, model, GPU, service or deployment operation was used.
