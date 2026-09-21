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

## Shared schemas

`-schema schema.json` accepts one described entity/classification schema. `-schemas schemas.json` accepts an ordered array of mixed entity, classification, relation and record groups. Groups share one encoder pass and one extraction candidate pool; classification choices do not become extraction queries.

```json
[
  {"parent":"entities","marker":"[E]","labels":["person","location"],
   "descriptions":[{"label":"person","text":"A human being"}]},
  {"parent":"lives_in","marker":"[R]","labels":["head","tail"]},
  {"parent":"person","marker":"[C]","labels":["name","city"],
   "record":{"mode":"natural","anchor_query_id":0,"fields":[
     {"query_id":0,"name":"name","scalar":true,"required":true},
     {"query_id":1,"name":"city","scalar":true}
   ]}}
]
```

`parent`, `marker` and ordered `labels` define a group. `[E]`, `[L]`, `[R]` and `[C]` select entities, classification, relations and records. Relation roles must be ordered `head`, `tail`. Record query IDs and the anchor are local to that group's labels, not global query indices. Duplicate task names remain separate groups in output.

Descriptions are an ordered list of `{ "label": "...", "text": "..." }`; examples are `{ "input": "...", "label": "..." }`. An optional `prompt` extends the parent text. Literal special tokens inside those strings do not create extra structural query slots. Schema files reject unknown fields, trailing data and invalid record metadata before loading weights.

## Compatibility limits

The implemented surface covers the published base checkpoint's native inference paths, not every upstream Python convenience API. Selection fields, custom validation callbacks, automatic document chunking, training/export of GLiNER checkpoints and remote Hub management are not included. Descriptions and examples are passed through the ordered schema format rather than the upstream Python schema-builder DSL.

The encoder supports the published DeBERTa configuration; incompatible architectures and nonzero query-attention layers are rejected. Non-default checkpoint variants need their own parity checks. Stable Go tie policies are deterministic but are not claimed to reproduce unspecified PyTorch top-k or SciPy assignment ties.

Record scoring has published-checkpoint parity for natural-anchor single and mixed fixtures, and synthetic PyTorch parity across all three modes. Exclusive decoding has algorithmic and brute-force assignment tests, not a full published-checkpoint fixture. The visual adapter work belongs to `model/jevlike`, not this package.

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
