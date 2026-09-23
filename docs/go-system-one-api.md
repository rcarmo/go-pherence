# TypeSafe question types

`POST /v1/systemone` accepts TypeSafe's core `state`/`questions` request and returns typed `noul`, `choice` and `score` answers. Gemma supplies the candidate probabilities. `/v1/decision` keeps its existing boolean/enum schema, prompts and batch contract.

## Request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"ticket": "Production is down and customers cannot connect."},
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does this need urgent handling?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle the ticket?",
        "criteria": {
          "operations": "Production outages",
          "billing": "Invoice and payment questions"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "Rate the incident severity.",
        "criteria": ["Routine request", "Degraded service", "Total outage"]
      }
    }
  }'
```

`model` is optional; when supplied it must match the loaded model ID. `state` is required and accepts text, a JSON object, an array or explicit `null`. Instructions are optional and accept the same types. Numbers and booleans may appear inside objects or arrays; they are not top-level entries.

| Type | Criteria | Answer |
|---|---|---|
| `noul` | Optional object containing descriptions for `true` and/or `false`; omitted or `null` is valid | `{"type":"noul","noul":p}` with `p` in `[0,1]`; no confidence field |
| `choice` | Object mapping 1–255 labels to descriptions | `type`, selected `choice`, `probabilities` keyed by label, `confidence` |
| `score` | Array of 2–255 levels, ordered lowest to highest | `type`, expected `score`, `probabilities` and `legend` keyed by zero-based level, `confidence` |

Descriptions accept text, objects, arrays or `null`. A null description leaves the label or level undescribed. It does not add a null answer. Noul is the probability of yes; it is not a boolean value or a severity score.

Question and choice insertion order is preserved during prompt construction. Exact probability ties select the first choice. Score levels retain array order, and `score = sum(i * p[i])`; a distribution `[0.2, 0.3, 0.5]` returns `1.3`, not the modal level `2`.

The response envelope contains `model`, `answers` keyed by question name, and `usage` with `input_tokens` and `output_tokens`. Input usage counts the rendered shared prompt and state. Output usage counts tokens in the serialised answer object; the model does not generate that JSON as a free-form continuation. Probabilities retain full precision and are normalised over the supplied candidates.

## Scoring and confidence

Noul scores the allowed values `false` and `true`, then returns `p(true)`. Choice scores the supplied labels using their descriptions. Score scores each ordered level and returns its expectation. All three use complete constrained token trees, including multi-token labels and score indices above nine. Greedy fallback cannot provide the required full distributions and is disabled on this route.

Confidence uses local formulas. For a choice with `K > 1` candidates it is `(max(p) - 1/K) / (1 - 1/K)`; a singleton has confidence 1. Score confidence is `1 - sum(p[i] * abs(i - mode)) / (K - 1)`, with the first modal level used on ties. Both are clamped to `[0,1]`.

TypeSafe describes confidence separately from candidate probabilities but does not publish the exact score formula. These formulas follow the public Kev approximation; numerical parity with TypeSafe confidence is unverified. Gemma probabilities are not calibrated correctness estimates. Thresholds need labelled validation for the intended task.

## Local limits

Requests are limited to 32 questions, 255 candidates per question and a 1 MiB body. Prompt-token and packed-memory limits match the existing decision engine. Both HTTP routes share one admission gate and return HTTP 429 while another request owns it. Invalid JSON, duplicate request/question/choice keys, unknown properties, unsupported types and forged tokenizer control markers are rejected before scoring. Rejected requests release the gate.

The TypeSafe playground update is a separate review slice. This adapter does not implement TypeSafe's hosted models, authentication, billing, `messages`, `options`, raw-logit diagnostics, `/permute`, `/separate` or batch extensions. A request has one `state`; use `/v1/decision` for its existing independent-context batch interface. Matching these request/answer types does not reproduce Jev's weights, training or calibration.

## Measured request costs

The [TypeSafe workload matrix](https://github.com/rcarmo/go-system-one/blob/efe3d929a9112fbfb6ad0ebfbf753d222e4c21dc/docs/benchmarks/README.md#typesafe-question-types) records five warm client HTTP samples per case at `594ba47`: 95.11 ms for Noul, 96.87 ms for Choice, 97.57 ms for Score and 130.57 ms for all three together. Each request evaluates one structured state. These are client HTTP intervals, not the `/v1/decision` handler times in the other charts.

## Port validation

This API-only port is independent of the packed-runtime hand-off (#22). Its released-model test runs the existing serial NVIDIA scorer. The standalone source also tested 512-row packing; those standalone results below are not a performance claim for this port.

## Contract and checks

The contract follows TypeSafe's [request][request], [entry][entry], [Noul][noul], [Choice][choice] and [Score][score] documentation, retrieved on 22 September 2026. Some third-party implementations differ: Simple Jev uses nine rating bins for Noul; this adapter follows TypeSafe's documented probability-of-yes semantics and uses two outcomes. Confidence references [`kev/api.py` at the reviewed revision][kev].

Offline tests cover structured/null entries, fractional scores, Noul endpoints and response shape, insertion-order ties, singleton choices, 255-level token trees, duplicate and malformed inputs, escaped control markers, shared admission and scalar/packed synthetic parity. Run them with:

```sh
go test ./model/gosystemone -run SystemOne -count=1
```

To run the released-model type check after verifying the [pinned artifacts](https://github.com/rcarmo/go-system-one/blob/ee905d0271a13569eda5449cfd03932e955337ca/docs/artifacts.md):

```sh
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B=/path/to/gemma-4-12b-it-UD-Q4_K_XL.gguf \
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER=/path/to/tokenizer \
go test ./model/gosystemone -run '^TestSystemOneReleasedModelTypes$' -v -count=1
```

The opt-in `TestSystemOneReleasedModelTypes` runs a mixed three-question request through the released tokenizer and NVIDIA scorer at serial and 512-row settings. Both passed; its saturated fixture had maximum score movement of about `0.00000681`. This checks types, distributions and execution, not task accuracy or a new numerical tolerance. Existing pinned llama.cpp decision and multi-field gates also pass unchanged.

[request]: https://docs.typesafe.ai/sdk/javascript/api/interfaces/SystemOneRequest
[entry]: https://docs.typesafe.ai/sdk/javascript/api/type-aliases/EntryType
[noul]: https://docs.typesafe.ai/sdk/javascript/api/interfaces/NoulResponse
[choice]: https://docs.typesafe.ai/sdk/javascript/api/interfaces/ChoiceResponse
[score]: https://docs.typesafe.ai/sdk/javascript/api/interfaces/ScoreResponse
[kev]: https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/api.py
