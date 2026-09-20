# Needle tool-call generation

Needle can generate a bounded array of tool calls using a schema-constrained token mask. **It never executes the calls.** A complete result means the output satisfies the supported grammar, not that the model chose the right tool, supplied truthful arguments, or has calibrated confidence.

The implementation is in [`model/needle`](../../model/needle/README.md). It uses the pinned Needle 3 training prompt layout, but is a Go implementation of a deliberately smaller schema language—not a port or compatibility claim for upstream's closed native grammar engine. The released Needle 3 archive has a CPU smoke test; Needle 2 tool-call quality has not been qualified.

## CLI

Create `tools.json`:

```json
[{"name":"light","description":"Set the light","parameters":{"type":"object","properties":{"on":{"type":"boolean"}},"required":["on"],"additionalProperties":false}}]
```

Create a UTF-8 query file, then run:

```sh
printf 'Turn the light on.' > query.txt
go run ./cmd/needle -model checkpoints/needle3/needle3.cact \
  -mode tools -tools tools.json -text-file query.txt -max-new 64
```

The released-model smoke prompt produced:

```json
{"function_calls":[{"name":"light","arguments":{"on":true}}],"json":"[{\"name\":\"light\",\"arguments\":{\"on\":true}}]","generated_ids":[507,379,300,1808,301,612,327,290,288,500,572],"complete":true,"executed":false,"calibrated":false}
```

`-system` adds an optional system instruction. `-max-calls` defaults to one (maximum four). Tools mode defaults to 128 output tokens; an explicit `-max-new` must be 1–1,024. It always uses cached decoding and stops at a complete JSON array rather than EOS. `-cached=false`, `-eos`, training/head controls and width slicing are rejected, not ignored. Depth slicing and opt-in hybrid `-packed` execution remain available, but the released tool-call smoke evidence uses the full-depth decoded model.

The CLI requires an original archive with a tokenizer containing the prompt markers. It rejects unknown top-level tool fields and trailing schema-file data. Errors produce no result JSON on stdout. It does not expose this path through either HTTP server or the embedded UI.

## Go API

```go
model, tokenizer, err := needle.LoadArchive("checkpoints/needle3/needle3.cact")
if err != nil { return err }
tools := []needle.ToolSchema{{
    Name: "light",
    Description: "Set the light",
    Parameters: json.RawMessage(`{"type":"object","properties":{"on":{"type":"boolean"}},"required":["on"]}`),
}}
result, err := model.GenerateTools(ctx, tokenizer, tools, "", "Turn the light on.",
    needle.ToolOptions{MaxNewTokens: 64, MaxCalls: 1})
if err != nil { return err } // Never use partial output as a completed call.
// result.Calls is data, not an instruction to execute it.
```

Import `context`, `encoding/json` and `github.com/rcarmo/go-pherence/model/needle` as needed. `ToolOptions.Decoder` controls the existing cache/work budgets and packed projections. `MaxBytes` defaults to 16 KiB and accepts 2–64 KiB; zero token/call/byte limits select API defaults (128/1/16 KiB). Negative values are errors.

`ToolPrompt` constructs the prompt without generating. `CompileToolGrammar`, `Start`, `Advance` and `Complete` expose immutable prefix recognition for the same grammar. `Advance` accepts raw token bytes, including incomplete UTF-8 sequences spanning tokens; it returns an owned state or rejects the continuation without changing the source state. `Tokenizer.PieceBytes` preserves those bytes, unlike whole-text decoding, and excludes control, unknown and user-defined marker tokens. The generation loop reuses private traversal scratch; public `Advance` allocates its own scratch.

## Accepted schema subset

This is a restricted generation language, not a general JSON Schema validator:

- Duplicate schema keys are rejected, including equivalent escaped key spellings; they cannot hide a prompt marker through last-key-wins decoding. Raw schema JSON nesting is limited to 32 before the stricter schema-depth check.
- Each tool has a unique name matching `[A-Za-z_][A-Za-z0-9_.-]*`, an optional description, and object-valued `parameters`.
- Objects emit properties in sorted-name order. **Every declared property must appear in `required`.** Optional, duplicate-required and unknown-required properties are rejected. `additionalProperties` must be `false` or omitted; omission still produces a closed object.
- Arrays require one `items` schema and an explicit `maxItems` of 1–8. `minItems` defaults to zero and must be between zero and `maxItems`.
- Scalars support `string`, `integer`, `number`, `boolean` (`bool` alias) and `null`. Scalar `const` and nonempty, type-compatible `enum` are supported. Object/array constants and mixed-type enums are not. `const` plus `enum` requires the constant's emitted literal to occur in the enum, with every member type-checked.
- `type` can be inferred from object/array keywords or scalar constants/enums. Type unions and conflicting keywords are errors. Integers use integer JSON spellings; enum/constant numbers retain the supplied spelling rather than accepting every numerically equivalent spelling.
- Strings admit valid UTF-8, JSON escapes and paired UTF-16 surrogate escapes. Invalid raw UTF-8 and unpaired generated surrogate escapes are rejected. Keys and constant strings use Go's canonical JSON string encoding, so not every equivalent escaped spelling is admitted.
- `title` and `description` are string-valued metadata, not validation constraints. All other unsupported keywords fail explicitly, including `$ref`, `$defs`, `$schema`, unions/combinators, `pattern`, `format`, `minimum`, `maximum`, string-length limits, `uniqueItems`, `default` and recursive schemas.

The outer array contains zero to `MaxCalls` entries; `[]` is a valid abstention. Each entry emits `name` followed by `arguments`. JSON whitespace is allowed. A tool may be selected more than once. There is no free-form reasoning output, optional-property enumeration, tool-result feedback loop or execution callback.

## Bounds and failure behavior

| Resource | Limit |
|---|---:|
| Tools / calls | 16 / 4 |
| Parameters per tool | 64 KiB |
| Combined parameters, names and descriptions | 128 KiB |
| One tool description | 4 KiB |
| Serialized prompt tool list / CLI schema file | 128 KiB each |
| Schema depth (parameter root is one) | 8 |
| Properties per object / array items | 32 / 8 |
| Generated regex text | 256 KiB |
| Expanded-program estimate / compiled instructions | 50,000 each |
| Query / system text | 32 KiB / 16 KiB |
| Output token surface / all candidate surfaces | 4 KiB / 4 MiB |
| Generated tokens / bytes | 1,024 / 64 KiB |
| Candidate grammar-work charge | 100,000,000 per request |

The expansion estimate is checked before regex simplification/compilation and is conservative; even schemas inside the structural limits can be rejected for expansion cost. The work charge is `(candidate byte length + 1) × program instruction count`, not a wall-clock guarantee. Model preparation, workspace and KV memory have their own decoder limits; these grammar limits do not make a released-model request small. Prompt plus requested generation must fit the model's bounded context, including its deployed archive window. There is no global-context eviction.

Candidates are considered in descending logit order, with token ID breaking ties. No sampling, tool code, shell command or network operation is involved. Context cancellation, incompatible tokenizers, unsupported schemas, context overflow, exhausted work/byte/token budgets and absence of a usable next token all return errors. API errors may include partial `JSON` and token IDs for diagnostics, but `Complete` remains false and `Calls` is unset. Partial output must never be promoted to a tool request.

Caller text and decoded schema strings/keys containing reserved chat/tool markers are rejected, including escaped marker spellings in schema JSON. This protects the prompt's structural delimiters; it does **not** make model-generated arguments safe to execute. Any separate application that consumes calls must perform its own authorization, validation and user-confirmation checks.

## Evidence and remaining work

See [the validation record](../validation/needle-tool-calls-20260920.md) for CPU smoke output, native ARM tests, grammar allocation measurements and regression gates. These checks establish bounded generation behavior, not general tool-use accuracy or calibration. Upstream native-engine equivalence, broader tool-use evaluation, HTTP/UI integration and Needle 2 tool quality remain unqualified.
