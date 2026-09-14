# Go transcription web app

`transcribe-web` is a trusted-LAN, no-login frontend for the durable Go speech pipeline. It supervises `speechjobserve` on private loopback and injects a random process-only bearer token into proxied API requests. Browsers never receive the token.

The Sigma deployment provides eight profiles: Auto, English, Portuguese and French for M4A and WAV. Jobs run serially in the durable queue. Interrupted work is requeued at startup. Successful jobs retain transcript/VTT and speaker artifacts while releasing the original upload and decoded PCM. Failed jobs retain media for explicit Retry. Cancellation and deletion remove media.

The app exposes no microphone, streaming, translation, named speaker identity, collaborative editing, cloud API or concurrent inference path.

## Build

```sh
export PATH=/var/home/agent/workspace/tools/go-toolchain/go/bin:$PATH
export GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2
go test ./cmd/audio/transcribe-web ./cmd/audio/speechjobserve ./runtime/speechjob ./runtime/speechjob/httpapi
go build -trimpath -o ~/.local/lib/transcribe-web/speechjobserve ./cmd/audio/speechjobserve
go build -trimpath -o ~/.local/lib/transcribe-web/transcribe-web ./cmd/audio/transcribe-web
```

## Deployment

The live config is administrator-generated at `~/.local/lib/transcribe-web/config.json`. It contains absolute SHA256-pinned model paths, eight immutable profiles, a private job store, durable queue, resource estimates, the expected Intel Iris Xe device and backend identity. Do not commit local paths or private job data.

Install the unit without starting it:

```sh
install -m 0644 cmd/audio/transcribe-web/deploy/transcribe-web.service ~/.config/systemd/user/transcribe-web.service
systemctl --user daemon-reload
systemctl --user status transcribe-web.service
```

Only start after an admitted trained-model/GPU window and metadata validation:

```sh
systemctl --user start transcribe-web.service
curl -fsS -H 'Host: sigma.local:8093' http://127.0.0.1:8093/
```

The LAN address is `http://sigma.local:8093`. There is no TLS or login. Anyone who can reach the port can upload, read, retry, cancel and delete jobs. Ports 8092 and 8701 and all legacy `whisper-stt` data remain separate.

## Verification limits

Model-free Go tests cover Auto language metadata, fixed-language byte compatibility, eight profile construction, queue recovery, no-login proxy boundaries, terminal media retention and child lifecycle helpers. Trained Whisper/Community inference, real browser flow, quality and performance require separate coordinated runs. `qualified:false` remains unchanged until those runs complete.
