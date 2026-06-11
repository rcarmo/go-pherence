package ideogram4

import (
	"fmt"
	"os"
	"strings"
)

func k3PrewarmEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_PREWARM")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3PrewarmAllLinears() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_PREWARM_ALL")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// PrewarmK3 prepares resident K3 FP8 linears. By default it only warms the DiT
// denoise linears; prewarming every Qwen/text/auxiliary linear can exceed memory
// on K3 when A100 Q8 caches are enabled. Set GO_PHERENCE_IDEOGRAM4_K3_PREWARM_ALL=1
// to restore the broader traversal for experiments.
func (m *DiTModel) PrewarmK3() int {
	if m == nil || !k3PrewarmEnabled() {
		return 0
	}
	count := 0
	pre := func(l *FP8Linear) {
		if k3FP8Prewarm(l) {
			count++
		}
	}
	if k3PrewarmAllLinears() {
		pre(m.Globals.LLMCondProj)
		pre(m.Globals.InputProj)
		pre(m.Globals.TimeIn)
		pre(m.Globals.TimeOut)
		pre(m.Globals.FinalAdaLN)
		pre(m.Globals.FinalLinear)
	}
	// These are the hot denoise path linears. They dominate first-use cost and are
	// safe to warm selectively for K3/A100 inference.
	pre(m.Globals.LLMCondProj)
	pre(m.Globals.InputProj)
	pre(m.Globals.TimeIn)
	pre(m.Globals.TimeOut)
	pre(m.Globals.FinalAdaLN)
	pre(m.Globals.FinalLinear)
	for i := range m.Layers {
		l := &m.Layers[i]
		pre(l.AdaLN)
		pre(l.QKV)
		pre(l.O)
		pre(l.W1)
		pre(l.W2)
		pre(l.W3)
	}
	return count
}

func (p *NativePipeline) PrewarmK3() int {
	if p == nil || !k3PrewarmEnabled() {
		return 0
	}
	return p.Conditioner.PrewarmK3() + p.Cond.PrewarmK3() + p.Uncond.PrewarmK3()
}

func (p *NativePipeline) maybePrewarmK3() {
	if !k3PrewarmEnabled() {
		return
	}
	n := p.PrewarmK3()
	if os.Getenv("GO_PHERENCE_IDEOGRAM4_TIMING") == "1" {
		fmt.Fprintf(os.Stderr, "timing k3_prewarm_linears=%d\n", n)
	}
}
