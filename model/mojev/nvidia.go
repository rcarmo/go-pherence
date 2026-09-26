package mojev

import (
	"context"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/internal/contextmutex"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

// Validate all host-side invariants before touching a device or packing weights.
func validateAcceleratedTextScorer(cpu *TextScorer, maxTokens int) error {
	if maxTokens < 3 || maxTokens > 512 || cpu == nil || cpu.head == nil || cpu.model == nil || len(cpu.norm) != 1024 || cpu.meta.VocabSize < 1 || len(cpu.embedding)%1024 != 0 || len(cpu.embedding)/1024 != cpu.meta.VocabSize || cpu.eps != 1e-6 || len(cpu.rope) < maxTokens*64 {
		return fmt.Errorf("mojev: invalid accelerated scorer configuration")
	}
	for _, v := range cpu.norm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("mojev: nonfinite final norm")
		}
	}
	return qwen.ValidateQwen35F32Branch(cpu.model, cpu.meta, maxTokens)
}

type gpuLayer struct {
	linear bool
	w      map[string]*nvidia.Buffer
}

// NVIDIATextScorer is an explicitly selected, bounded resident PTX backend for
// the repaired F32 text policy. Calls are serialised over reusable GPU scratch;
// no candidate KV/recurrent state survives a branch. Close waits for active work.
// The CPU scorer supplies immutable embeddings and the F32 head readout; the
// uploaded encoder is not retained. The caller's CPU scorer remains usable.
type NVIDIATextScorer struct {
	mu                    contextmutex.Mutex
	cpu                   *TextScorer
	maxTokens             int
	layers                []gpuLayer
	norm                  *nvidia.Buffer
	scratch               map[string]*nvidia.Buffer
	kernels               map[string]nvidia.CUfunction
	module                *nvidia.PTXModule
	buffers               []*nvidia.Buffer
	resident              int64
	closed                bool
	commands              []nvidia.KernelLaunch
	commandCount          int
	hostInput, hostOutput []float32
	tree                  *nvidia.Buffer
	treeRows              []uint32
}

// NewNVIDIATextScorer uploads text weights once. maxTokens bounds each candidate
// path, not a packed request. This backend requires NVIDIA compute capability
// compatible with the embedded sm_86 PTX and keeps the CPU implementation intact.
// If construction and cleanup both fail, the returned closed scorer is non-nil
// solely so the caller can retry Close; it cannot be used for inference.
func NewNVIDIATextScorer(cpu *TextScorer, maxTokens int) (result *NVIDIATextScorer, err error) {
	if err := validateAcceleratedTextScorer(cpu, maxTokens); err != nil {
		return nil, err
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
	// Keep only immutable host data still used after upload. Retaining cpu
	// itself would unnecessarily keep its ~2GB encoder layers alive. This view
	// neither mutates the caller's scorer nor copies the embedding/head payload.
	host := &TextScorer{head: cpu.head, embedding: cpu.embedding, meta: cpu.meta, eps: cpu.eps}
	g := &NVIDIATextScorer{cpu: host, maxTokens: maxTokens, scratch: map[string]*nvidia.Buffer{}, kernels: map[string]nvidia.CUfunction{}}
	ok := false
	defer func() {
		if !ok {
			if cleanupErr := g.Close(); cleanupErr != nil {
				result, err = g, errors.Join(err, cleanupErr)
			}
		}
	}()
	names := []string{"mj_gemm", "mj_norm", "mj_add", "mj_silu_mul", "mj_conv", "mj_l2", "mj_delta", "mj_gated_norm", "mj_qk_norm_rope", "mj_attention", "mj_tree_conv", "mj_tree_delta", "mj_tree_qk_norm_rope", "mj_tree_attention"}
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
	g.tree, err = g.alloc(maxTokens * 4)
	if err != nil {
		return nil, err
	}
	g.treeRows = make([]uint32, maxTokens*4)
	g.commands = make([]nvidia.KernelLaunch, 512)
	g.hostInput = make([]float32, maxTokens*1024)
	g.hostOutput = make([]float32, maxTokens*1024)
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

// Close waits for active work and releases the module and device buffers. A
// driver failure closes inference immediately but retains resources for a later
// Close retry; callers must check its error before shutting down the runtime.
// Successful close also drops the host embedding/head references.
func (g *NVIDIATextScorer) Close() error {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closed = true
	if g.module != nil {
		// PTXModule.Close synchronises before unloading. This also drains any
		// prefix launched before a mid-batch failure; never free buffers first.
		if err := g.module.Close(); err != nil {
			return err
		}
		g.module = nil
	}
	for i := len(g.buffers) - 1; i >= 0; i-- {
		g.buffers[i].Free()
	}
	g.buffers = nil
	g.layers = nil
	g.norm = nil
	g.scratch = nil
	g.kernels = nil
	g.commands = nil
	g.hostInput, g.hostOutput = nil, nil
	g.tree, g.treeRows = nil, nil
	g.cpu = nil
	g.resident = 0
	return nil
}
func (g *NVIDIATextScorer) launch(name string, blocks int, args ...unsafe.Pointer) error {
	return nvidia.LaunchKernel(g.kernels[name], uint32(blocks), 1, 1, 256, 1, 1, 0, args...)
}

// queue records by-value arguments into preallocated stable launch storage.
func (g *NVIDIATextScorer) queue(name string, gx, gy int, args ...uint64) error {
	if g.commandCount >= len(g.commands) || len(args) > 8 {
		return fmt.Errorf("mojev: GPU command capacity")
	}
	c := &g.commands[g.commandCount]
	c.Function = g.kernels[name]
	c.Grid = [3]uint32{uint32(gx), uint32(gy), 1}
	c.Block = [3]uint32{256, 1, 1}
	if name == "mj_gemm" {
		c.Block[0] = 128
	}
	c.ArgCount = len(args)
	copy(c.Args[:], args)
	g.commandCount++
	return nil
}

// The branch kernel signatures lack the final tree-metadata argument.
func (g *NVIDIATextScorer) queueTree(name string, blocks int, tree bool, args ...uint64) error {
	if !tree {
		args = args[:len(args)-1]
	}
	return g.queue(name, blocks, 1, args...)
}

func (g *NVIDIATextScorer) normRows(x, w, y *nvidia.Buffer, rows, dim int, zero int32) error {
	return g.queue("mj_norm", rows, 1, uint64(x.Ptr), uint64(w.Ptr), uint64(y.Ptr), uint64(rows), uint64(dim), uint64(math.Float32bits(g.cpu.eps)), uint64(zero))
}
func (g *NVIDIATextScorer) binary(name string, x, y *nvidia.Buffer, n int) error {
	return g.queue(name, (n+255)/256, 1, uint64(x.Ptr), uint64(y.Ptr), uint64(n))
}
func (g *NVIDIATextScorer) project(x, w, y *nvidia.Buffer, rows, in, out int) error {
	if g.commands != nil {
		return g.queue("mj_gemm", (out+63)/64, (rows+31)/32, uint64(x.Ptr), uint64(w.Ptr), uint64(y.Ptr), uint64(rows), uint64(out), uint64(in))
	}
	m, n, k := int32(rows), int32(out), int32(in)
	return nvidia.LaunchKernel(g.kernels["mj_gemm"], uint32((out+63)/64), uint32((rows+31)/32), 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&y.Ptr), unsafe.Pointer(&m), unsafe.Pointer(&n), unsafe.Pointer(&k))
}

func (g *NVIDIATextScorer) encodeBranch(b TextBranch) ([]float32, error) {
	return g.encodeTree(b, nil)
}

func (g *NVIDIATextScorer) encodeTree(b TextBranch, ends []int) ([]float32, error) {
	return g.encodeTreeContext(context.Background(), b, ends)
}

func (g *NVIDIATextScorer) encodeTreeContext(ctx context.Context, b TextBranch, ends []int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n := len(b.IDs)
	if n > g.maxTokens {
		return nil, fmt.Errorf("mojev: branch exceeds GPU token capacity %d", g.maxTokens)
	}
	if ends != nil {
		if err := fillGPUTree(g.treeRows[:n*4], n, b.StateLen, b.QuestionLen, ends); err != nil {
			return nil, err
		}
		if err := g.tree.UploadUint32(g.treeRows[:n*4]); err != nil {
			return nil, err
		}
	}
	input := g.hostInput[:n*1024]
	g.commandCount = 0
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
	r, eps := uint64(n), uint64(math.Float32bits(g.cpu.eps))
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
			convName, deltaName := "mj_conv", "mj_delta"
			if ends != nil {
				convName, deltaName = "mj_tree_conv", "mj_tree_delta"
			}
			if e := g.queueTree(convName, (n*6144+255)/256, ends != nil, uint64(qkv.Ptr), uint64(cw.Ptr), uint64(conv.Ptr), r, uint64(g.tree.Ptr)); e != nil {
				return nil, e
			}
			if e := g.queue("mj_l2", n*32, 1, uint64(conv.Ptr), r, eps); e != nil {
				return nil, e
			}
			a, dt, alpha, beta, out := w["a"], w["dt"], s["alpha"], s["beta"], s["attn"]
			if e := g.queueTree(deltaName, 256, ends != nil, uint64(conv.Ptr), uint64(alpha.Ptr), uint64(beta.Ptr), uint64(dt.Ptr), uint64(a.Ptr), uint64(out.Ptr), r, uint64(g.tree.Ptr)); e != nil {
				return nil, e
			}
			z, gn := s["z"], w["gate_norm"]
			if e := g.queue("mj_gated_norm", n*16, 1, uint64(out.Ptr), uint64(z.Ptr), uint64(gn.Ptr), r, eps); e != nil {
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
			ropeName := "mj_qk_norm_rope"
			if ends != nil {
				ropeName = "mj_tree_qk_norm_rope"
			}
			for _, p := range []struct {
				name, weight              string
				heads, stride, headStride int32
			}{{"qg", "qn", 8, 4096, 512}, {"k", "kn", 2, 512, 256}} {
				buf, weight := s[p.name], w[p.weight]
				if e := g.queueTree(ropeName, n*int(p.heads), ends != nil, uint64(buf.Ptr), uint64(weight.Ptr), r, uint64(p.heads), uint64(p.stride), uint64(p.headStride), eps, uint64(g.tree.Ptr)); e != nil {
					return nil, e
				}
			}
			q, k, v, out := s["qg"], s["k"], s["v"], s["attn"]
			attentionName, arg1, arg2 := "mj_attention", uint64(b.StateLen), uint64(b.QuestionLen)
			if ends != nil {
				attentionName, arg1, arg2 = "mj_tree_attention", uint64(g.tree.Ptr), uint64(b.StateLen+b.QuestionLen)
			}
			if e := g.queue(attentionName, n*8, 1, uint64(q.Ptr), uint64(k.Ptr), uint64(v.Ptr), uint64(out.Ptr), r, arg1, arg2); e != nil {
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
	if e := g.launchContext(ctx); e != nil {
		return nil, e
	}
	out := g.hostOutput[:n*1024]
	if e := norm.Download(out); e != nil {
		return nil, e
	}
	return out, nil
}

// ScoreEncoded executes repaired text inference on resident PTX weights.
func (g *NVIDIATextScorer) ScoreEncoded(row EncodedRow) ([][]float32, error) {
	return g.ScoreEncodedContext(context.Background(), row)
}

// ScoreEncodedContext cancels waits and drains any launched GPU work before
// returning a context error. Already running kernels cannot be preempted.
func (g *NVIDIATextScorer) ScoreEncodedContext(ctx context.Context, row EncodedRow) ([][]float32, error) {
	if g == nil {
		return nil, fmt.Errorf("mojev: nil GPU scorer")
	}
	if err := g.mu.LockContext(ctx); err != nil {
		return nil, err
	}
	defer g.mu.Unlock()
	if g.closed || g.cpu == nil || g.cpu.head == nil || g.module == nil {
		return nil, fmt.Errorf("mojev: GPU scorer closed or uninitialized")
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
	return g.scoreTree(ctx, row)
}

// ScoreText includes the same validation, tokenizer and public answer path as CPU.
func (g *NVIDIATextScorer) ScoreText(req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int) (*TextDecision, error) {
	return g.ScoreTextContext(context.Background(), req, tok, stateLimit, questionLimit)
}

// ScoreTextContext is the cancellable tokenizer, scorer and answer path.
func (g *NVIDIATextScorer) ScoreTextContext(ctx context.Context, req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int) (*TextDecision, error) {
	if g == nil {
		return nil, fmt.Errorf("mojev: nil GPU scorer")
	}
	if err := g.mu.LockContext(ctx); err != nil {
		return nil, err
	}
	ready := !g.closed && g.cpu != nil
	g.mu.Unlock()
	// Preprocessing needs no weights. If Close wins before inference, the
	// locked ScoreEncodedContext callback rejects the request transactionally.
	if !ready {
		return nil, fmt.Errorf("mojev: GPU scorer closed or uninitialized")
	}
	return scoreTextContextWith(ctx, req, tok, stateLimit, questionLimit, func(row EncodedRow) ([][]float32, error) { return g.ScoreEncodedContext(ctx, row) })
}
