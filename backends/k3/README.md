# backends/k3

High-level **compute-backend dispatch** for the SpaceMIT K3 SoC (MilkV
Jupiter 2): backend selection, op placement, and dispatch across the available
runtimes.

| File | Role |
|---|---|
| `backend.go` | Backend interface + runtime-priority stack |
| `select.go` | Backend selection / capability probing |
| `ops.go` | Op dispatch |
| `spacemit.go` | SpaceMIT ORT / AI-core path |
| `vulkan.go` | Vulkan (PowerVR BXM-4-64) path |
| `simd.go` | Portable SIMD fallback wiring |

> This is the **dispatch layer**, distinct from `backends/spacemit/k3engine`,
> which is the pure-Go inference *engine*. They share the "k3" name (the SoC) but
> sit at different levels: `backends/k3` chooses *where* compute runs;
> `k3engine` *is* one of the pure-Go compute paths.
