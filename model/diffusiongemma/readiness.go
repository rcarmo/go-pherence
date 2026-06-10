package diffusiongemma

import "strings"

// ReadinessSummary is a compact, user-facing runtime status derived from
// capabilities, shard availability, and tensor readiness.
type ReadinessSummary struct {
	TextScaffoldReady bool     `json:"text_scaffold_ready"`
	ShardsReady       bool     `json:"shards_ready"`
	ReferenceComplete bool     `json:"reference_complete"`
	RuntimeReady      bool     `json:"runtime_ready"`
	Missing           []string `json:"missing,omitempty"`
}

func BuildReadinessSummary(caps RuntimeCapabilities, shards *ShardAvailability, readiness *TensorReadiness) ReadinessSummary {
	out := ReadinessSummary{TextScaffoldReady: caps.TextOnlyScaffoldReady, ReferenceComplete: caps.ReferenceComplete, RuntimeReady: caps.RuntimeReady}
	if shards != nil {
		out.ShardsReady = shards.Ready
	}
	if readiness != nil && !readiness.TextReady {
		out.Missing = append(out.Missing, "text tensor readiness")
	}
	if shards == nil || !shards.Ready {
		out.Missing = append(out.Missing, "safetensor shards")
	}
	out.Missing = append(out.Missing, caps.MissingForReference...)
	return out
}

func (s ReadinessSummary) String() string {
	parts := []string{
		"text_scaffold=" + boolString(s.TextScaffoldReady),
		"shards=" + boolString(s.ShardsReady),
		"reference_complete=" + boolString(s.ReferenceComplete),
		"runtime_ready=" + boolString(s.RuntimeReady),
	}
	if len(s.Missing) > 0 {
		parts = append(parts, "missing="+strings.Join(s.Missing, ","))
	}
	return strings.Join(parts, " ")
}

func boolString(v bool) string {
	if v {
		return "true"
	}
	return "false"
}
