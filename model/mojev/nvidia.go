package mojev

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

type gpuLayer struct {
	linear bool
	w      map[string]*nvidia.Buffer
}

// NVIDIATextScorer is an explicitly selected, bounded resident PTX backend for
// the repaired F32 text policy. Calls are serialised over reusable GPU scratch;
// no candidate KV/recurrent state survives a branch. Close waits for active work.
// The CPU scorer supplies immutable embeddings and the F32 head readout.
type NVIDIATextScorer struct {
	mu        sync.Mutex
	cpu       *TextScorer
	maxTokens int
	layers    []gpuLayer
	norm      *nvidia.Buffer
	scratch   map[string]*nvidia.Buffer
	kernels   map[string]nvidia.CUfunction
	module    *nvidia.PTXModule
	buffers   []*nvidia.Buffer
	resident  int64
	closed    bool
}

// NewNVIDIATextScorer uploads text weights once. maxTokens bounds each candidate
// path, not a packed request. This backend requires NVIDIA compute capability
// compatible with the embedded sm_86 PTX and keeps the CPU implementation intact.
func NewNVIDIATextScorer(cpu *TextScorer, maxTokens int) (*NVIDIATextScorer, error) {
	if cpu == nil || cpu.model == nil || cpu.head == nil || len(cpu.model.Layers) != 24 || cpu.meta.HiddenSize != 1024 || maxTokens < 3 || maxTokens > 512 {
		return nil, fmt.Errorf("mojev: invalid NVIDIA scorer configuration")
	}
	if !nvidia.Init() {
		return nil, fmt.Errorf("mojev: NVIDIA unavailable")
	}
	major, minor := nvidia.ComputeCapability()
	if major < 8 || (major == 8 && minor < 6) {
		return nil, fmt.Errorf("mojev: PTX requires compute capability 8.6 or later")
	}
	// F32 text layers plus conservative scratch reserve. Never evict other users.
	free, _ := nvidia.MemInfo()
	if free < 4<<30 {
		return nil, fmt.Errorf("mojev: insufficient free GPU memory (need 4 GiB headroom)")
	}
	g := &NVIDIATextScorer{cpu: cpu, maxTokens: maxTokens, scratch: map[string]*nvidia.Buffer{}, kernels: map[string]nvidia.CUfunction{}}
	ok := false
	defer func() {
		if !ok {
			g.Close()
		}
	}()
	names := []string{"mj_gemm", "mj_norm", "mj_add", "mj_silu_mul", "mj_conv", "mj_l2", "mj_delta", "mj_gated_norm", "mj_qk_norm_rope", "mj_attention"}
	module, err := nvidia.LoadPTXFunctions(ptx.MoJev, names)
	if err != nil {
		return nil, err
	}
	g.module = module
	for _, name := range names {
		g.kernels[name] = module.Function(name)
	}
	upload := func(data []float32, in, out int) (*nvidia.Buffer, error) {
		if in > 0 {
			if len(data) != in*out {
				return nil, fmt.Errorf("mojev: invalid GPU weight shape")
			}
			transposed := make([]float32, len(data))
			for o := 0; o < out; o++ {
				for i := 0; i < in; i++ {
					transposed[i*out+o] = data[o*in+i]
				}
			}
			data = transposed
		}
		b, e := g.alloc(len(data))
		if e != nil {
			return nil, e
		}
		if e = b.Upload(data); e != nil {
			return nil, e
		}
		return b, nil
	}
	for _, layer := range cpu.model.Layers {
		w := map[string]*nvidia.Buffer{}
		linear := layer.Kind == qwen.Qwen35LinearAttentionLayerKind
		type entry struct {
			name    string
			data    []float32
			in, out int
		}
		var entries []entry
		if linear {
			l := layer.Linear
			entries = []entry{{"in", l.InputNorm.Data(), 0, 0}, {"post", l.PostNorm.Data(), 0, 0}, {"qkv", l.QKVW.Data(), 1024, 6144}, {"z", l.GateW.Data(), 1024, 2048}, {"conv", l.Conv1D.Data(), 0, 0}, {"a", l.A.Data(), 0, 0}, {"dt", l.DTBias.Data(), 0, 0}, {"alpha", l.AlphaW.Data(), 1024, 16}, {"beta", l.BetaW.Data(), 1024, 16}, {"gate_norm", l.Norm.Data(), 0, 0}, {"o", l.OutW.Data(), 2048, 1024}, {"gate", l.MLPGateW.Data(), 1024, 3584}, {"up", l.MLPUpW.Data(), 1024, 3584}, {"down", l.MLPDownW.Data(), 3584, 1024}}
		} else {
			l := layer.Full
			entries = []entry{{"in", l.InputNorm.Data(), 0, 0}, {"post", l.PostNorm.Data(), 0, 0}, {"qg", l.QW.Data(), 1024, 4096}, {"k", l.KW.Data(), 1024, 512}, {"v", l.VW.Data(), 1024, 512}, {"qn", l.QNorm.Data(), 0, 0}, {"kn", l.KNorm.Data(), 0, 0}, {"o", l.OW.Data(), 2048, 1024}, {"gate", l.GateW.Data(), 1024, 3584}, {"up", l.UpW.Data(), 1024, 3584}, {"down", l.DownW.Data(), 3584, 1024}}
		}
		for _, e := range entries {
			b, err := upload(e.data, e.in, e.out)
			if err != nil {
				return nil, err
			}
			w[e.name] = b
		}
		g.layers = append(g.layers, gpuLayer{linear, w})
	}
	g.norm, err = upload(cpu.norm, 0, 0)
	if err != nil {
		return nil, err
	}
	for name, width := range map[string]int{"x": 1024, "norm": 1024, "qkv": 6144, "conv": 6144, "z": 2048, "alpha": 16, "beta": 16, "attn": 2048, "qg": 4096, "k": 512, "v": 512, "proj": 1024, "gate": 3584, "up": 3584} {
		g.scratch[name], err = g.alloc(maxTokens * width)
		if err != nil {
			return nil, err
		}
	}
	ok = true
	return g, nil
}
func (g *NVIDIATextScorer) alloc(n int) (*nvidia.Buffer, error) {
	b, e := nvidia.Malloc(n)
	if e != nil {
		return nil, e
	}
	g.buffers = append(g.buffers, b)
	g.resident += int64(b.Size)
	return b, nil
}
func (g *NVIDIATextScorer) ResidentBytes() int64 {
	if g == nil {
		return 0
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.resident
}
func (g *NVIDIATextScorer) Close() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	_ = nvidia.SyncErr()
	for i := len(g.buffers) - 1; i >= 0; i-- {
		g.buffers[i].Free()
	}
	g.buffers = nil
	g.resident = 0
	if g.module != nil {
		_ = g.module.Close()
		g.module = nil
	}
	g.closed = true
}
func (g *NVIDIATextScorer) launch(name string, blocks int, args ...unsafe.Pointer) error {
	return nvidia.LaunchKernel(g.kernels[name], uint32(blocks), 1, 1, 256, 1, 1, 0, args...)
}
func (g *NVIDIATextScorer) normRows(x, w, y *nvidia.Buffer, rows, dim int, zero int32) error {
	r, d, e := int32(rows), int32(dim), g.cpu.eps
	return g.launch("mj_norm", rows, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&y.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&d), unsafe.Pointer(&e), unsafe.Pointer(&zero))
}
func (g *NVIDIATextScorer) binary(name string, x, y *nvidia.Buffer, n int) error {
	size := int32(n)
	return g.launch(name, (n+255)/256, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&y.Ptr), unsafe.Pointer(&size))
}
func (g *NVIDIATextScorer) project(x, w, y *nvidia.Buffer, rows, in, out int) error {
	m, n, k := int32(rows), int32(out), int32(in)
	return nvidia.LaunchKernel(g.kernels["mj_gemm"], uint32((out+63)/64), uint32((rows+31)/32), 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&y.Ptr), unsafe.Pointer(&m), unsafe.Pointer(&n), unsafe.Pointer(&k))
}

func (g *NVIDIATextScorer) encodeBranch(b TextBranch) ([]float32, error) {
	n := len(b.IDs)
	if n > g.maxTokens {
		return nil, fmt.Errorf("mojev: branch exceeds GPU token capacity %d", g.maxTokens)
	}
	input := make([]float32, n*1024)
	for i, id := range b.IDs {
		if id < 0 || id >= g.cpu.meta.VocabSize {
			return nil, fmt.Errorf("mojev: token outside vocabulary")
		}
		copy(input[i*1024:], g.cpu.embedding[id*1024:(id+1)*1024])
	}
	s := g.scratch
	x, norm := s["x"], s["norm"]
	if err := x.Upload(input); err != nil {
		return nil, err
	}
	r, eps := int32(n), g.cpu.eps
	for _, l := range g.layers {
		w := l.w
		if err := g.normRows(x, w["in"], norm, n, 1024, 1); err != nil {
			return nil, err
		}
		if l.linear {
			for _, p := range []struct {
				name  string
				width int
			}{{"qkv", 6144}, {"z", 2048}, {"alpha", 16}, {"beta", 16}} {
				if e := g.project(norm, w[p.name], s[p.name], n, 1024, p.width); e != nil {
					return nil, e
				}
			}
			qkv, conv, cw := s["qkv"], s["conv"], w["conv"]
			if e := g.launch("mj_conv", (n*6144+255)/256, unsafe.Pointer(&qkv.Ptr), unsafe.Pointer(&cw.Ptr), unsafe.Pointer(&conv.Ptr), unsafe.Pointer(&r)); e != nil {
				return nil, e
			}
			if e := g.launch("mj_l2", n*32, unsafe.Pointer(&conv.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&eps)); e != nil {
				return nil, e
			}
			a, dt, alpha, beta, out := w["a"], w["dt"], s["alpha"], s["beta"], s["attn"]
			if e := g.launch("mj_delta", 256, unsafe.Pointer(&conv.Ptr), unsafe.Pointer(&alpha.Ptr), unsafe.Pointer(&beta.Ptr), unsafe.Pointer(&dt.Ptr), unsafe.Pointer(&a.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&r)); e != nil {
				return nil, e
			}
			z, gn := s["z"], w["gate_norm"]
			if e := g.launch("mj_gated_norm", n*16, unsafe.Pointer(&out.Ptr), unsafe.Pointer(&z.Ptr), unsafe.Pointer(&gn.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&eps)); e != nil {
				return nil, e
			}
		} else {
			for _, p := range []struct {
				name  string
				width int
			}{{"qg", 4096}, {"k", 512}, {"v", 512}} {
				if e := g.project(norm, w[p.name], s[p.name], n, 1024, p.width); e != nil {
					return nil, e
				}
			}
			for _, p := range []struct {
				name, weight              string
				heads, stride, headStride int32
			}{{"qg", "qn", 8, 4096, 512}, {"k", "kn", 2, 512, 256}} {
				buf, weight := s[p.name], w[p.weight]
				if e := g.launch("mj_qk_norm_rope", n*int(p.heads), unsafe.Pointer(&buf.Ptr), unsafe.Pointer(&weight.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&p.heads), unsafe.Pointer(&p.stride), unsafe.Pointer(&p.headStride), unsafe.Pointer(&eps)); e != nil {
					return nil, e
				}
			}
			q, k, v, out := s["qg"], s["k"], s["v"], s["attn"]
			ns, nq := int32(b.StateLen), int32(b.QuestionLen)
			if e := g.launch("mj_attention", n*8, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&ns), unsafe.Pointer(&nq)); e != nil {
				return nil, e
			}
		}
		if e := g.project(s["attn"], w["o"], s["proj"], n, 2048, 1024); e != nil {
			return nil, e
		}
		if e := g.binary("mj_add", x, s["proj"], n*1024); e != nil {
			return nil, e
		}
		if e := g.normRows(x, w["post"], norm, n, 1024, 1); e != nil {
			return nil, e
		}
		if e := g.project(norm, w["gate"], s["gate"], n, 1024, 3584); e != nil {
			return nil, e
		}
		if e := g.project(norm, w["up"], s["up"], n, 1024, 3584); e != nil {
			return nil, e
		}
		if e := g.binary("mj_silu_mul", s["gate"], s["up"], n*3584); e != nil {
			return nil, e
		}
		if e := g.project(s["gate"], w["down"], s["proj"], n, 3584, 1024); e != nil {
			return nil, e
		}
		if e := g.binary("mj_add", x, s["proj"], n*1024); e != nil {
			return nil, e
		}
	}
	if e := g.normRows(x, g.norm, norm, n, 1024, 1); e != nil {
		return nil, e
	}
	if e := nvidia.SyncErr(); e != nil {
		return nil, e
	}
	out := make([]float32, n*1024)
	if e := norm.Download(out); e != nil {
		return nil, e
	}
	return out, nil
}

// ScoreEncoded executes repaired text inference on resident PTX weights.
func (g *NVIDIATextScorer) ScoreEncoded(row EncodedRow) ([][]float32, error) {
	if g == nil {
		return nil, fmt.Errorf("mojev: nil GPU scorer")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil, fmt.Errorf("mojev: GPU scorer closed")
	}
	// Reject over-capacity branches before launching any work.
	for f, q := range row.Questions {
		if f >= len(row.Candidates) {
			return nil, fmt.Errorf("mojev: inconsistent candidates")
		}
		for _, c := range row.Candidates[f] {
			if len(row.State) > g.maxTokens || len(q) > g.maxTokens-len(row.State) || len(c) > g.maxTokens-len(row.State)-len(q) {
				return nil, fmt.Errorf("mojev: branch exceeds GPU token capacity")
			}
		}
	}
	return ScoreBranchLocalText(row, g.cpu.head, g.encodeBranch)
}

// ScoreText includes the same validation, tokenizer and public answer path as CPU.
func (g *NVIDIATextScorer) ScoreText(req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int) (*TextDecision, error) {
	if g == nil {
		return nil, fmt.Errorf("mojev: nil GPU scorer")
	}
	return g.cpu.scoreText(req, tok, stateLimit, questionLimit, g.ScoreEncoded)
}
