package ideogram4

// ReleaseGPU frees cached GPU-resident resources owned by DiT globals. It is a
// no-op unless opt-in GPU FP8 caching has uploaded one or more linears.
func (g *DiTGlobals) ReleaseGPU() {
	if g == nil {
		return
	}
	for _, lin := range []*FP8Linear{g.InputProj, g.LLMCondProj, g.TimeIn, g.TimeOut, g.AdaLNProj, g.FinalAdaLN, g.FinalLinear} {
		lin.ReleaseGPU()
	}
}

// ReleaseGPU frees cached GPU-resident resources owned by one DiT layer.
func (l *DiTLayer) ReleaseGPU() {
	if l == nil {
		return
	}
	if l.gpu != nil {
		l.gpu.Free()
		l.gpu = nil
	}
	for _, lin := range []*FP8Linear{l.QKV, l.O, l.W1, l.W2, l.W3, l.AdaLN} {
		lin.ReleaseGPU()
	}
}

// ReleaseGPU frees cached GPU-resident FP8 linears owned by the DiT model.
func (m *DiTModel) ReleaseGPU() {
	if m == nil {
		return
	}
	m.Globals.ReleaseGPU()
	for i := range m.Layers {
		m.Layers[i].ReleaseGPU()
	}
}

// ReleaseGPU frees cached GPU-resident resources owned by the full native
// pipeline. VAE and Qwen conditioner currently use transient NVIDIA buffers,
// while DiT FP8 linears may hold cached weights; this method is safe to call
// unconditionally at phase boundaries.
func (p *NativePipeline) ReleaseGPU() {
	if p == nil {
		return
	}
	p.Cond.ReleaseGPU()
	p.Uncond.ReleaseGPU()
}
