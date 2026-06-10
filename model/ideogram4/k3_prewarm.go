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

// PrewarmK3 decodes/packs K3-resident representations for long-lived FP8
// linears. On non-riscv64 or when GO_PHERENCE_IDEOGRAM4_K3 is unset this is a
// no-op. The current K3 bridge uses resident fp16 rows; future IME2 packers can
// attach their own resident buffers behind the same FP8Linear hook.
func (m *DiTModel) PrewarmK3() int {
	if m == nil {
		return 0
	}
	count := 0
	pre := func(l *FP8Linear) {
		if k3FP8Prewarm(l) {
			count++
		}
	}
	if m.Globals.InputProj != nil {
		pre(m.Globals.InputProj)
		pre(m.Globals.LLMCondProj)
		pre(m.Globals.TimeIn)
		pre(m.Globals.TimeOut)
		pre(m.Globals.AdaLNProj)
		pre(m.Globals.FinalAdaLN)
		pre(m.Globals.FinalLinear)
	}
	for i := range m.Layers {
		pre(m.Layers[i].QKV)
		pre(m.Layers[i].O)
		pre(m.Layers[i].W1)
		pre(m.Layers[i].W2)
		pre(m.Layers[i].W3)
		pre(m.Layers[i].AdaLN)
	}
	return count
}

func (p *NativePipeline) PrewarmK3() int {
	if p == nil {
		return 0
	}
	return p.Cond.PrewarmK3() + p.Uncond.PrewarmK3()
}

func (p *NativePipeline) maybePrewarmK3() {
	if !k3PrewarmEnabled() {
		return
	}
	if n := p.PrewarmK3(); n > 0 && os.Getenv("GO_PHERENCE_IDEOGRAM4_TIMING") == "1" {
		fmt.Fprintf(os.Stderr, "timing k3_prewarm_linears=%d\n", n)
	}
}
