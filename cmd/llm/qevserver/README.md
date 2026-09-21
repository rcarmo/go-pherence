# qevserver

`qevserver` serves finite boolean and enum decisions from the QEV v1 Gemma 4 12B checkpoint.

```sh
go run ./cmd/llm/qevserver \
  -model /path/to/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  -tokenizer-dir /path/to/gemma-4-12b-it-tokenizer \
  -backend nvidia \
  -listen 127.0.0.1:8080
```

Open `http://127.0.0.1:8080/qev` for the standalone playground. The API endpoint is `POST /v1/decision`.

The command verifies the frozen model, tokenizer, tokenizer configuration and chat-template SHA-256 values before loading them. Use `-verify-artifacts=false` only for development fixtures. `-backend simd` selects the correctness-oracle implementation; the 12B SIMD path is too slow for interactive use.

The server has no authentication or TLS. Bind it to loopback or put an authenticated reverse proxy in front of all routes.

See [`docs/validation/qev-v1-20260921.md`](../../../docs/validation/qev-v1-20260921.md) for pins, numerical gates and measured warm decision latency.
