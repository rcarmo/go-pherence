## Native GLiNER 2.5 inference

The native path loads the `fastino/gliner2.5-base-v1` safetensors checkpoint and runs DeBERTa, boundary encoding, shared candidate pooling and reranking in Go. Dense batch projections use checked Plan 9 SGEMM on supported CPUs; dot products use the SIMD runtime. Unsupported CPUs retain scalar fallbacks.

The local model directory needs `config.json`, `encoder_config/config.json`, `tokenizer.json` and `model.safetensors`. The CLI does not download weights.

```sh
go run ./cmd/gliner2 -model /path/to/model \
  -text 'Ada Lovelace lived in London.' -label person -label location

go run ./cmd/gliner2 -model /path/to/model \
  -text 'The service was excellent.' -classify sentiment \
  -label positive -label negative -label neutral

go run ./cmd/gliner2 -model /path/to/model \
  -text 'Ada Lovelace lived in London.' -relation lives_in

go run ./cmd/gliner2 -model /path/to/model \
  -text 'Ada Lovelace lived in London.' -record person \
  -field name:str,required -field city:str -anchor name
```

Labels and fields retain command-line order. Classification returns independent sigmoid probabilities, not a softmax distribution. Entity decoding applies checkpoint temperature, abstention and optional count guidance before resolving overlaps per label. Relation output includes mention consolidation and exact source offsets.

Record fields accept `str` or `list`, followed by `,required` and/or `,exclusive`. Modes are `natural`, `latent` and `anchorless`. Natural mode defaults to the first field as anchor; the others reject `-anchor`. Exclusive scalars use global assignment, while exclusive lists give each candidate one owner. Required lists follow upstream's threshold behaviour and do not fabricate a value when nothing passes.

`-raw` exposes model scores for entity, relation and record workflows. `-max-tokens` rejects over-budget combined schema/text inputs rather than silently truncating their alignment. Byte offsets slice Go strings; character offsets count Unicode code points.

## Compatibility limits

The CLI runs one entity group, classification task, relation type or record group per request. Mixed task schemas, multiple record/relation groups, schema descriptions, few-shot examples, selection fields, custom validators and automatic document chunking are not implemented. This is not a drop-in replacement for the complete upstream Python API.

The encoder supports the published DeBERTa configuration; incompatible architectures and nonzero query-attention layers are rejected. Non-default checkpoint variants need their own parity checks. Stable Go tie policies are deterministic but are not claimed to reproduce unspecified PyTorch top-k or SciPy assignment ties.

Record scoring has published-checkpoint parity for one natural-anchor fixture and synthetic PyTorch parity across all three modes. Exclusive decoding has algorithmic tests, not a full published-checkpoint fixture. The visual adapter work belongs to `model/jevlike`, not this package.

## Checking results

Ordinary tests contain tiny Transformers and PyTorch fixtures, tokenizer examples, scalar references and decoding regressions. The full checkpoint is external:

```sh
go test ./model/gliner2 ./cmd/gliner2
go vet ./model/gliner2 ./cmd/gliner2
GLINER_MODEL_DIR=/path/to/model \
GLINER_TOKENIZER_JSON=/path/to/model/tokenizer.json \
  go test ./model/gliner2 -run Published -v -count=1
```

Recorded timings and parity scope are in [VALIDATION.md](VALIDATION.md). Upstream attribution is in [PROVENANCE.md](PROVENANCE.md).
