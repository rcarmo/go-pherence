# go-system-one

`go-system-one` serves finite boolean and enum decisions from the Go System One v1 Gemma 4 12B checkpoint.

```sh
go run ./cmd/llm/go-system-one \
  -model /path/to/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  -tokenizer-dir /path/to/gemma-4-12b-it-tokenizer \
  -backend nvidia \
  -listen 127.0.0.1:8080
```

Open `http://127.0.0.1:8080/go-system-one` for the standalone playground. The API endpoint is `POST /v1/decision`. Tree-mode results display every allowed outcome and its constrained model probability; the selected outcome is highlighted. These are model probabilities over the allowed candidates, not calibrated correctness estimates.

![Go System One decision playground](../../../docs/images/go-system-one-playground.png)

The command verifies the frozen model, tokenizer, tokenizer configuration and chat-template SHA-256 values before loading them. Use `-verify-artifacts=false` only for development fixtures. `-backend simd` selects the correctness-oracle implementation; the 12B SIMD path is too slow for interactive use.

## Deployment limits

The server has no authentication or TLS. Its default loopback binding is the supported direct-use configuration. For remote access, put an authenticated TLS reverse proxy in front of every route, restrict the upstream to loopback or a private socket, and apply request-rate and body-size limits at the proxy. Do not expose the command directly to an untrusted network.

Inference uses single admission. A concurrent request receives HTTP 429 with `decision inference busy; retry later`; clients should use bounded backoff rather than parallel retries. The command verifies model artifacts by default, accepts at most a 1 MiB JSON body and shuts down with a 30-second drain window. The NVIDIA configuration is qualified for one RTX 3060 12 GB process with 9,173,277,696 resident projection bytes; co-locating another model or inference worker requires separate VRAM admission measurements.

The endpoint returns constrained model probabilities. They are not calibrated accuracy estimates. Log requests and decisions according to the deployment's data-handling policy, and avoid placing sensitive context text in proxy access logs.

See [`docs/validation/go-system-one-v1-20260921.md`](../../../docs/validation/go-system-one-v1-20260921.md) for pins, numerical gates, workload measurements and retention evidence.
