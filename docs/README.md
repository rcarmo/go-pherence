# go-pherence documentation

Start with the task you want to run. Model inspection, native inference and numerical parity are different levels of support; a successful config load does not imply a working generator.

| Task | Read |
|---|---|
| Run a CLI or service | [Command index](guides/commands.md) |
| Choose a model or checkpoint format | [Supported models](models/supported-models.md) |
| Select a backend or adjust placement | [Backends](backends/README.md) and [runtime tuning](guides/tuning.md) |
| Understand package ownership | [Architecture](architecture/README.md) |
| Transcribe, translate or label speakers | [Speech](speech/README.md) |
| Score variable choices or train a scorer | [Jevlike](../model/jevlike/README.md) |
| Extract entities, relations, records or classes | [GLiNER 2.5](../model/gliner2/README.md) |
| Run block-diffusion text generation | [DiffusionGemma](models/diffusiongemma/README.md) |
| Test changes on this host or cross-build | [Validation gates](validation/validation-gates.md) |
| Review cross-repository safety findings and limits | [Safety audit](validation/repository-safety-audit-20260919.md), [package-review closeout](validation/repository-safety-audit-closeout-20260920.md), [host remediation](validation/repository-safety-host-remediation-20260920.md), [open findings/platform issues](validation/repository-safety-audit-open-findings-20260920.md) and [package coverage](validation/repository-safety-audit-coverage-20260919.md) |
| Compare measured workloads | [Performance](performance/README.md) |
| Investigate an older result or design decision | [History](history/README.md) |

## Where things live

* `guides/` contains commands, asset setup and tuning.
* `models/` describes model-specific support boundaries. Jevlike and GLiNER keep their API, validation and provenance documents beside their implementation in `model/`.
* `speech/`, `backends/` and `architecture/` cover their respective runtime layers.
* `validation/` defines checks; `performance/` links measurements to workload contracts.
* `history/` retains implementation logs, investigations and superseded plans. Their numbers are evidence of the recorded run, not a promise about the current tree.

Machine-readable coverage and speech manifests stay at the `docs/` root because tools consume those paths. The generated [model coverage snapshot](model-coverage-snapshot.md) tracks a limited family set; it is not a repository-wide support percentage. The [coverage tracker](models/model-coverage-status.md) explains the gates.

## Diagrams

![Core LLM and NVIDIA execution path](architecture.svg)

![Focused backend validation matrix](test-matrix.svg)

Run `make docs-diagrams` to regenerate these from `scripts/render-architecture.ts` and `scripts/render-test-matrix.ts`. The test diagram shows a selected matrix, not proof that the full suite passes. `make docs-check` checks generated diagrams and local links, including new Markdown files.
