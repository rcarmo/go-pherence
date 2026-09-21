# Go transcription web app

`transcribe-web` is a trusted-LAN, no-login frontend for the durable Go speech pipeline. It supervises `speechjobserve` on private loopback and injects a random process-only bearer token into proxied API requests. Browsers never receive the token.

The Sigma deployment provides sixteen profiles: ASR-only and ASR plus speaker labels for Auto, English, Portuguese and French, each with M4A and WAV input. All profiles share one loaded Whisper model; diarization profiles also share one loaded Community-1 model. Jobs run serially in the durable queue. The browser shows active work above a persistent recording library. Recording titles are derived from source filenames and can be edited without changing the original filename or checkpoint identity. Interrupted work is requeued at startup. Successful jobs retain transcript artifacts while releasing the original upload and decoded PCM. Failed jobs retain media for explicit Retry. Cancellation and deletion remove media.

Conversational profiles set Community-1 `tie_policy` to `lowest-index`. Equal reconstruction scores are resolved by stable speaker index and listed in the diarization checkpoint as ambiguous frames. Speaker labels stay experimental. Use `reject` only for diagnostics that must stop at the first equal-score cutoff.

The browser checks WAV and M4A container signatures before upload. A filename/content mismatch is blocked with conversion guidance. The API exposes only allowlisted failure codes; raw stage errors and local paths stay private. Permanent media-type failures do not offer Retry because the retained bytes cannot succeed under the same profile. Primary downloads prefer speaker-labelled VTT when available. Additional JSON and VTT variants are under Export options. Export filenames use the recording title and explicit language, for example `customer-interview.pt.speakers.vtt`; internal job IDs are excluded.

The app exposes no microphone, streaming, translation, named speaker identity, collaborative editing, cloud API or concurrent inference path.

## Build

```sh
export PATH=/var/home/agent/workspace/tools/go-toolchain/go/bin:$PATH
export GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2
go test ./cmd/audio/transcribe-web ./cmd/audio/speechjobserve ./runtime/speechjob ./runtime/speechjob/httpapi
go build -trimpath -o ~/.local/lib/transcribe-web/speechjobserve ./cmd/audio/speechjobserve
go build -trimpath -o ~/.local/lib/transcribe-web/transcribe-web ./cmd/audio/transcribe-web
```

### Standalone frontend binary

`transcribe-web` embeds `web/index.html`, `web/app.js` and `web/style.css` through `//go:embed web/*`. Build it from the repository root with the required Go version:

```sh
mkdir -p dist
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
  go build -trimpath -buildvcs=true -ldflags='-s -w' \
  -o dist/transcribe-web ./cmd/audio/transcribe-web
```

The resulting executable needs no `web/` directory or shared C libraries at runtime. Check the build before distribution:

```sh
file dist/transcribe-web
ldd dist/transcribe-web                 # expected: not a dynamic executable
go version -m dist/transcribe-web       # expected: CGO_ENABLED=0
go test ./cmd/audio/transcribe-web      # reads the embedded assets
sha256sum dist/transcribe-web
```

The embedded files are fixed at build time. Rebuild the executable after changing anything under `cmd/audio/transcribe-web/web/`.

This executable contains the trusted-LAN frontend and reverse proxy. It still starts a separate `speechjobserve` executable and requires an administrator-generated JSON configuration. Model weights, model configuration, tokenizer files, Community-1 assets, the job store and queue are external files. Keep their absolute paths in the JSON configuration; do not embed private recordings, credentials or model files.

Run the frontend with absolute child and configuration paths:

```sh
./dist/transcribe-web \
  --listen '[::]:8093' \
  --hosts 'sigma.local:8093,127.0.0.1:8093,localhost:8093' \
  --backend 'http://127.0.0.1:18093' \
  --speechjobserve "$HOME/.local/lib/transcribe-web/speechjobserve" \
  --config "$HOME/.local/lib/transcribe-web/config.json"
```

`--backend` must use a literal loopback address. `--speechjobserve` and `--config` must name existing absolute regular files; the child executable must have an execute bit.

## Deployment

The live config is administrator-generated at `~/.local/lib/transcribe-web/config.json`. It contains absolute SHA256-pinned model paths, sixteen immutable profiles, a private job store, durable queue, resource estimates, the expected Intel Iris Xe device and backend identity. Do not commit local paths or private job data.

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

Model-free Go tests cover Auto language metadata, fixed-language byte compatibility, sixteen shared-model profile pipelines, queue recovery, no-login proxy boundaries, terminal media retention and child lifecycle helpers. Trained Whisper/Community inference, real browser flow, quality and performance require separate coordinated runs. `qualified:false` remains unchanged until those runs complete.
