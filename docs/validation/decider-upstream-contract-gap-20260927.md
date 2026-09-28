# Decider upstream contract gap after v1.5 audit

The native Go `model/decider` port preserves the released 0.8B state-first scorer pinned to upstream source `c4daaac28af9fea95d627015cffa2dd5a5926ee6`. Current Apache-2.0 upstream `Mapika/decider` at `a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f` (v1.5.0) changed the **public answer contract** after that pin. Existing Go `SystemOne` responses should not silently adopt those changes.

| Surface | Go pinned contract | Current upstream |
|---|---|---|
| Choice `confidence` | Rounded largest option probability | Rounded `(n*p_max - 1)/(n - 1)`, clipped to `[0,1]`; uniform is 0 |
| Score `confidence` | Rounded largest level probability | Rounded `1 - expected_distance_from_modal_level / mean_uniform_distance_from_middle`, clipped to `[0,1]` |
| `x_p_max` | Absent | Separately reports rounded largest probability for Choice/Score |
| Noul with absent instructions | Rejected | May use `"Which answer fits the context?"` when a true/false criterion has a description |
| Per-answer-type temperature | Optional state-first map in Go as of `a938a2f6` | State-first and schema-cache maps, overrides and serving reporting |

Upstream commit `6b9c2818d429ace74172a36aebf2351c0a1188bb` introduced TypeSafe confidence and the Noul instruction fallback; `421c3f0` tightened Noul criteria shape. Upstream tests in `tests/test_systemone.py` and `tests/test_serve_http.py` check the new confidence and `x_p_max` fields. The Go port's pinned source uses the old maximum-probability `confidence`, so changing its existing JSON field would alter consumers' interpretation even when the logits and selected answer stay identical.

A bounded compatibility implementation should expose a separately named or explicitly versioned answer format, replay upstream synthetic probability/validation fixtures, and keep the old `SystemOne` output unchanged. A released checkpoint is not required to test these output formulas. Serving, schema-first caching, vLLM, 2B/4B/35B and vision models need separate design and approval. Model-free formula compatibility is not a calibration-quality result; held-out labels are needed to fit and assess temperatures or changed confidence thresholds.
