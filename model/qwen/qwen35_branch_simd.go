package qwen

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	cfg "github.com/rcarmo/go-pherence/loader/config"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
	"github.com/rcarmo/go-pherence/tensor"
)

// Qwen35SIMDBranch is a bounded, prepacked dense-F32 branch executor. Shared
// weights are immutable; one mutex serialises reusable scratch. The causal
// generation APIs and scalar branch reference are unchanged.
type Qwen35SIMDBranch struct {
	mu        sync.Mutex
	model     *Qwen35BaseModel
	meta      cfg.QwenNativeMTPMetadata
	maxTokens int
	packed    map[*tensor.Tensor][]float32
	scratch   map[string][]float32
}

func NewQwen35SIMDBranch(m *Qwen35BaseModel, meta cfg.QwenNativeMTPMetadata, maxTokens int) (*Qwen35SIMDBranch, error) {
	if m == nil || len(m.Layers) != 24 || len(m.Layers) != meta.MainLayerCount() || meta.BF16Trajectory || !meta.ZeroCenteredRMSNorm || meta.HiddenSize != 1024 || meta.IntermediateSize != 3584 || meta.NumAttentionHeads != 8 || meta.NumKeyValueHeads != 2 || meta.HeadDim != 256 || meta.LinearNumKeyHeads != 16 || meta.LinearNumValueHeads != 16 || meta.LinearKeyHeadDim != 128 || meta.LinearValueHeadDim != 128 || meta.LinearConvKernelDim != 4 || meta.PartialRotaryFactor != 0.25 || maxTokens < 3 || maxTokens > 512 {
		return nil, fmt.Errorf("qwen: unsupported SIMD branch configuration")
	}
	out := &Qwen35SIMDBranch{model: m, meta: meta, maxTokens: maxTokens, packed: map[*tensor.Tensor][]float32{}, scratch: map[string][]float32{}}
	pack := func(t *tensor.Tensor, in, n int) error {
		if t == nil || len(t.Data()) != in*n {
			return fmt.Errorf("qwen: missing dense SIMD tensor")
		}
		for _, v := range t.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("qwen: nonfinite SIMD tensor")
			}
		}
		p, e := simd.PackSgemmNTWeights(t.Data(), n, in, in)
		if e != nil {
			return e
		}
		out.packed[t] = p
		return nil
	}
	for idx, layer := range m.Layers {
		want := Qwen35LinearAttentionLayerKind
		if (idx+1)%4 == 0 {
			want = Qwen35FullAttentionLayerKind
		}
		if layer.Kind != want {
			return nil, fmt.Errorf("qwen: unexpected SIMD layer %d", idx)
		}
		type weight struct {
			t     *tensor.Tensor
			in, n int
		}
		var ws []weight
		if layer.Kind == Qwen35LinearAttentionLayerKind {
			l := layer.Linear
			if e := ValidateQwen35LinearAttentionLayer(l, meta, "SIMD"); e != nil {
				return nil, e
			}
			ws = []weight{{l.QKVW, 1024, 6144}, {l.GateW, 1024, 2048}, {l.AlphaW, 1024, 16}, {l.BetaW, 1024, 16}, {l.OutW, 2048, 1024}, {l.MLPGateW, 1024, 3584}, {l.MLPUpW, 1024, 3584}, {l.MLPDownW, 3584, 1024}}
		} else if layer.Kind == Qwen35FullAttentionLayerKind {
			l := layer.Full
			if e := ValidateQwen35FullAttentionLayer(l, meta, "SIMD"); e != nil {
				return nil, e
			}
			ws = []weight{{l.QW, 1024, 4096}, {l.KW, 1024, 512}, {l.VW, 1024, 512}, {l.OW, 2048, 1024}, {l.GateW, 1024, 3584}, {l.UpW, 1024, 3584}, {l.DownW, 3584, 1024}}
		} else {
			return nil, fmt.Errorf("qwen: invalid SIMD layer kind")
		}
		for _, w := range ws {
			if e := pack(w.t, w.in, w.n); e != nil {
				return nil, e
			}
		}
	}
	out.scratch["ssm"] = make([]float32, 16*128*128)
	// 12 is a common multiple of the 6-row amd64 and 4-row ARM64/RVV
	// microtiles. Pad projections only, never attention or recurrent tokens.
	out.scratch["padIn"] = make([]float32, ((maxTokens+11)/12)*12*3584)
	out.scratch["padOut"] = make([]float32, ((maxTokens+11)/12)*12*6144)
	for name, width := range map[string]int{"x": 1024, "norm": 1024, "qkv": 6144, "conv": 6144, "z": 2048, "alpha": 16, "beta": 16, "attn": 2048, "qg": 4096, "q": 2048, "gateq": 2048, "k": 512, "v": 512, "proj": 1024, "gate": 3584, "up": 3584} {
		out.scratch[name] = make([]float32, maxTokens*width)
	}
	return out, nil
}
func (s *Qwen35SIMDBranch) project(dst, x []float32, w *tensor.Tensor, rows, in, out int) error {
	if rows%12 != 0 && s.scratch != nil {
		padded := (rows + 11) / 12 * 12
		input, output := s.scratch["padIn"][:padded*in], s.scratch["padOut"][:padded*out]
		copy(input, x[:rows*in])
		clear(input[rows*in:])
		if err := s.projectRows(output, input, w, padded, in, out); err != nil {
			return err
		}
		copy(dst, output[:rows*out])
		return nil
	}
	return s.projectRows(dst, x, w, rows, in, out)
}

func (s *Qwen35SIMDBranch) projectRows(dst, x []float32, w *tensor.Tensor, rows, in, out int) error {
	clear(dst[:rows*out])
	workers := min(runtime.GOMAXPROCS(0), 6, out/16)
	if workers < 2 || out < 512 {
		if !simd.SgemmNTPrepackedTo(dst, x, w.Data(), s.packed[w], rows, out, in, 1, in, in, out) {
			return fmt.Errorf("qwen: SIMD projection failed")
		}
		return nil
	}
	var wg sync.WaitGroup
	errs := make([]error, workers)
	for worker := 0; worker < workers; worker++ {
		start := (out / 16) * worker / workers * 16
		end := (out / 16) * (worker + 1) / workers * 16
		wg.Add(1)
		go func(i, start, end int) {
			defer wg.Done()
			if !simd.SgemmNTPrepackedTo(dst[start:], x, w.Data()[start*in:], s.packed[w][start*in:end*in], rows, end-start, in, 1, in, in, out) {
				errs[i] = fmt.Errorf("qwen: SIMD projection failed")
			}
		}(worker, start, end)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// Forward returns owned pre-final-normalisation hidden rows. All positions are
// branch-local and recurrence starts empty; unrelated nodes are absent.
func (s *Qwen35SIMDBranch) Forward(inputs [][]float32, ns, nq int, rope []float32, eps float32) ([][]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("qwen: nil SIMD branch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(inputs)
	if n < 3 || n > s.maxTokens || ns < 1 || ns >= n || nq < 1 || nq >= n-ns || eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) || len(rope) < n*64 {
		return nil, fmt.Errorf("qwen: invalid SIMD branch inputs")
	}
	b := func(name string, width int) []float32 { return s.scratch[name][:n*width] }
	x, norm := b("x", 1024), b("norm", 1024)
	for t, row := range inputs {
		if len(row) != 1024 {
			return nil, fmt.Errorf("qwen: invalid SIMD input row")
		}
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("qwen: nonfinite SIMD input")
			}
		}
		copy(x[t*1024:], row)
	}
	normalise := func(w []float32) {
		copy(norm, x)
		for t := 0; t < n; t++ {
			rmsNormQwen35InPlace(norm[t*1024:(t+1)*1024], w, eps, true)
		}
	}
	for _, layer := range s.model.Layers {
		var post, ow, gatew, upw, downw *tensor.Tensor
		attn := b("attn", 2048)
		if layer.Kind == Qwen35LinearAttentionLayerKind {
			l := layer.Linear
			normalise(l.InputNorm.Data())
			post, ow, gatew, upw, downw = l.PostNorm, l.OutW, l.MLPGateW, l.MLPUpW, l.MLPDownW
			qkv, z, alpha, beta := b("qkv", 6144), b("z", 2048), b("alpha", 16), b("beta", 16)
			for _, p := range []struct {
				dst []float32
				w   *tensor.Tensor
				out int
			}{{qkv, l.QKVW, 6144}, {z, l.GateW, 2048}, {alpha, l.AlphaW, 16}, {beta, l.BetaW, 16}} {
				if e := s.project(p.dst, norm, p.w, n, 1024, p.out); e != nil {
					return nil, e
				}
			}
			conv := b("conv", 6144)
			clear(conv)
			cw := l.Conv1D.Data()
			for t := 0; t < n; t++ {
				row := conv[t*6144 : (t+1)*6144]
				for k := 0; k < 4; k++ {
					p := t - 3 + k
					if p >= 0 {
						src := qkv[p*6144 : (p+1)*6144]
						weight := cw[k*6144 : (k+1)*6144]
						for c := range row {
							row[c] += src[c] * weight[c]
						}
					}
				}
				siluInPlace(row)
				for h := 0; h < 32; h++ {
					l2NormalizeInPlace(row[h*128:(h+1)*128], eps)
				}
			}
			state := s.scratch["ssm"]
			clear(state)
			for t := 0; t < n; t++ {
				q, k, v := conv[t*6144:t*6144+2048], conv[t*6144+2048:t*6144+4096], conv[t*6144+4096:(t+1)*6144]
				out := attn[t*2048 : (t+1)*2048]
				for h := 0; h < 16; h++ {
					decay := float32(math.Exp(float64(softplus(alpha[t*16+h]+l.DTBias.Data()[h]) * l.A.Data()[h])))
					be := sigmoid(beta[t*16+h])
					qh, kh := q[h*128:(h+1)*128], k[h*128:(h+1)*128]
					for j := 0; j < 128; j++ {
						row := state[(h*128+j)*128 : (h*128+j+1)*128]
						simd.VecScale(row, row, decay)
						memory := simd.Sdot(row, kh)
						simd.Saxpy((v[h*128+j]-memory)*be, kh, row)
						out[h*128+j] = simd.Sdot(row, qh) * float32(1/math.Sqrt(128))
					}
				}
				if e := qwen35GatedRMSNormValueHeads(out, z[t*2048:(t+1)*2048], l.Norm.Data(), 16, 128, eps); e != nil {
					return nil, e
				}
			}
		} else {
			l := layer.Full
			normalise(l.InputNorm.Data())
			post, ow, gatew, upw, downw = l.PostNorm, l.OW, l.GateW, l.UpW, l.DownW
			qg, k, v := b("qg", 4096), b("k", 512), b("v", 512)
			for _, p := range []struct {
				dst []float32
				w   *tensor.Tensor
				out int
			}{{qg, l.QW, 4096}, {k, l.KW, 512}, {v, l.VW, 512}} {
				if e := s.project(p.dst, norm, p.w, n, 1024, p.out); e != nil {
					return nil, e
				}
			}
			q, gates := b("q", 2048), b("gateq", 2048)
			for t := 0; t < n; t++ {
				for h := 0; h < 8; h++ {
					copy(q[t*2048+h*256:], qg[t*4096+h*512:t*4096+h*512+256])
					copy(gates[t*2048+h*256:], qg[t*4096+h*512+256:t*4096+(h+1)*512])
					rmsNormQwen35InPlace(q[t*2048+h*256:t*2048+(h+1)*256], l.QNorm.Data(), eps, true)
				}
				for h := 0; h < 2; h++ {
					rmsNormQwen35InPlace(k[t*512+h*256:t*512+(h+1)*256], l.KNorm.Data(), eps, true)
				}
				llmops.ApplyRoPEPartial(q[t*2048:(t+1)*2048], rope, t, 8, 256, 32)
				llmops.ApplyRoPEPartial(k[t*512:(t+1)*512], rope, t, 2, 256, 32)
			}
			for t := 0; t < n; t++ {
				end := n
				if t < ns {
					end = ns
				} else if t < ns+nq {
					end = ns + nq
				}
				o := qwenMTPGroupedAttention(q[t*2048:(t+1)*2048], k[:end*512], v[:end*512], 8, 2, 256)
				for j := range o {
					attn[t*2048+j] = o[j] * sigmoid(gates[t*2048+j])
				}
			}
		}
		proj := b("proj", 1024)
		if e := s.project(proj, attn, ow, n, 2048, 1024); e != nil {
			return nil, e
		}
		for i := range x {
			x[i] += proj[i]
		}
		normalise(post.Data())
		gate, up := b("gate", 3584), b("up", 3584)
		if e := s.project(gate, norm, gatew, n, 1024, 3584); e != nil {
			return nil, e
		}
		if e := s.project(up, norm, upw, n, 1024, 3584); e != nil {
			return nil, e
		}
		for i := range gate {
			gate[i] = gate[i] * sigmoid(gate[i]) * up[i]
		}
		if e := s.project(proj, gate, downw, n, 3584, 1024); e != nil {
			return nil, e
		}
		for i := range x {
			x[i] += proj[i]
		}
	}
	out := make([][]float32, n)
	for t := range out {
		for _, v := range x[t*1024 : (t+1)*1024] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("qwen: nonfinite SIMD output")
			}
		}
		out[t] = append([]float32(nil), x[t*1024:(t+1)*1024]...)
	}
	return out, nil
}
