package minicpmv

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// ResamplerCPU owns the F32 one-layer perceiver-resampler weights. Older
// MiniCPM-V checkpoints persist a query-grid position tensor; every supported
// variant regenerates the upstream source-grid 2D sin/cos table.
type ResamplerCPU struct {
	query, position []float32
	kvProj          []float32
	inProjWeight    []float32
	inProjBias      []float32
	outProj         visionLinear
	lnQWeight       []float32
	lnQBias         []float32
	lnKVWeight      []float32
	lnKVBias        []float32
	lnPostWeight    []float32
	lnPostBias      []float32
	finalProj       []float32
	numQuery        int
	embedDim        int
	kvDim           int
	heads           int
	headDim         int
}

func LoadResamplerCPUFromDir(dir string, cfg config.MiniCPMVConfig) (*ResamplerCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadResamplerCPU(src, cfg)
}

func LoadResamplerCPU(src Float32TensorSource, cfg config.MiniCPMVConfig) (*ResamplerCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O resampler tensor source")
	}
	s := cfg.MiniCPMVSummary()
	shape, err := NewResamplerShape(s.NumQuery, s.HiddenSize, s.ResamplerHeads, s.VisionHiddenSize)
	if err != nil {
		return nil, err
	}
	m := &ResamplerCPU{numQuery: shape.NumQuery, embedDim: shape.EmbedDim, kvDim: shape.KVDim, heads: shape.NumHeads, headDim: shape.EmbedDim / shape.NumHeads}
	if m.kvDim <= 0 {
		m.kvDim = m.embedDim
	}
	if m.embedDim%4 != 0 {
		return nil, fmt.Errorf("MiniCPM-V/O resampler embed_dim=%d must be divisible by 4 for 2D positions", m.embedDim)
	}
	if m.query, err = loadTextTensor(src, "resampler.query", []int{m.numQuery, m.embedDim}); err != nil {
		return nil, err
	}
	if shape.NeedsKVProjection {
		if m.kvProj, err = loadTextTensor(src, "resampler.kv_proj.weight", []int{m.embedDim, m.kvDim}); err != nil {
			return nil, err
		}
	}
	if m.inProjWeight, err = loadTextTensor(src, "resampler.attn.in_proj_weight", []int{3 * m.embedDim, m.embedDim}); err != nil {
		return nil, err
	}
	if m.inProjBias, err = loadTextTensor(src, "resampler.attn.in_proj_bias", []int{3 * m.embedDim}); err != nil {
		return nil, err
	}
	if m.outProj, err = loadVisionLinear(src, "resampler.attn.out_proj", m.embedDim, m.embedDim); err != nil {
		return nil, err
	}
	if m.lnQWeight, err = loadTextTensor(src, "resampler.ln_q.weight", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.lnQBias, err = loadTextTensor(src, "resampler.ln_q.bias", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.lnKVWeight, err = loadTextTensor(src, "resampler.ln_kv.weight", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.lnKVBias, err = loadTextTensor(src, "resampler.ln_kv.bias", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.lnPostWeight, err = loadTextTensor(src, "resampler.ln_post.weight", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.lnPostBias, err = loadTextTensor(src, "resampler.ln_post.bias", []int{m.embedDim}); err != nil {
		return nil, err
	}
	if m.finalProj, err = loadTextTensor(src, "resampler.proj", []int{m.embedDim, m.embedDim}); err != nil {
		return nil, err
	}
	// MiniCPM-V 2.0 persists query-grid positions. MiniCPM-O/V 2.6 derives
	// dynamic source-grid positions and has no tensor in its checkpoint.
	if data, tensorShape, getErr := src.GetFloat32("resampler.pos_embed"); getErr == nil {
		if !equalTextShape(tensorShape, []int{m.numQuery, m.embedDim}) || len(data) != m.numQuery*m.embedDim {
			return nil, fmt.Errorf("load resampler.pos_embed: shape=%v want [%d %d]", tensorShape, m.numQuery, m.embedDim)
		}
		if err := validateFiniteTextOutput("resampler position", data); err != nil {
			return nil, err
		}
		m.position = data
	} else if !IsLikelySigLIPVision(s) {
		return nil, fmt.Errorf("load resampler.pos_embed: %w", getErr)
	}
	return m, nil
}

func (m *ResamplerCPU) Resample(visionTokens []float32, visionTokensCount, visionHidden int) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O CPU resampler")
	}
	if visionTokensCount <= 0 || visionHidden != m.kvDim {
		return nil, fmt.Errorf("invalid MiniCPM-V/O resampler input tokens=%d hidden=%d want hidden=%d", visionTokensCount, visionHidden, m.kvDim)
	}
	values, ok := checked.MulInt(visionTokensCount, visionHidden)
	if !ok || len(visionTokens) != values {
		return nil, fmt.Errorf("invalid MiniCPM-V/O resampler input values=%d want %d", len(visionTokens), values)
	}
	grid := int(math.Sqrt(float64(visionTokensCount)))
	if grid*grid != visionTokensCount {
		return nil, fmt.Errorf("MiniCPM-V/O resampler token count=%d is not a square patch grid", visionTokensCount)
	}
	if err := validateFiniteTextOutput("resampler input", visionTokens); err != nil {
		return nil, err
	}
	kv := make([]float32, visionTokensCount*m.embedDim)
	if len(m.kvProj) == 0 {
		copy(kv, visionTokens)
	} else {
		for token := 0; token < visionTokensCount; token++ {
			if !simd.GemvRows(kv[token*m.embedDim:(token+1)*m.embedDim], visionTokens[token*m.kvDim:(token+1)*m.kvDim], m.kvProj, m.embedDim, m.kvDim) {
				return nil, fmt.Errorf("MiniCPM-V/O resampler KV projection failed")
			}
		}
	}
	normKV := make([]float32, len(kv))
	if !simd.LayerNormLastAxisTo(normKV, kv, visionTokensCount, m.embedDim, m.lnKVWeight, m.lnKVBias, 1e-6) {
		return nil, fmt.Errorf("MiniCPM-V/O resampler KV LayerNorm failed")
	}
	normQ := make([]float32, len(m.query))
	if !simd.LayerNormLastAxisTo(normQ, m.query, m.numQuery, m.embedDim, m.lnQWeight, m.lnQBias, 1e-6) {
		return nil, fmt.Errorf("MiniCPM-V/O resampler query LayerNorm failed")
	}
	if len(m.position) != 0 {
		if len(m.position) != len(normQ) || !simd.VecAddTo(normQ, normQ, m.position) {
			return nil, fmt.Errorf("MiniCPM-V/O resampler query position addition failed")
		}
	}
	positions := resampler2DPositions(grid, grid, m.embedDim)
	if len(positions) != visionTokensCount*m.embedDim {
		return nil, fmt.Errorf("MiniCPM-V/O resampler position generation failed")
	}
	keysInput := make([]float32, len(normKV))
	if !simd.VecAddTo(keysInput, normKV, positions) {
		return nil, fmt.Errorf("MiniCPM-V/O resampler position addition failed")
	}
	q, k, v := make([]float32, len(normQ)), make([]float32, len(normKV)), make([]float32, len(normKV))
	qWeight := m.inProjWeight[:m.embedDim*m.embedDim]
	kWeight := m.inProjWeight[m.embedDim*m.embedDim : 2*m.embedDim*m.embedDim]
	vWeight := m.inProjWeight[2*m.embedDim*m.embedDim:]
	qBias := m.inProjBias[:m.embedDim]
	kBias := m.inProjBias[m.embedDim : 2*m.embedDim]
	vBias := m.inProjBias[2*m.embedDim:]
	for query := 0; query < m.numQuery; query++ {
		if err := packedResamplerProjection(q[query*m.embedDim:(query+1)*m.embedDim], normQ[query*m.embedDim:(query+1)*m.embedDim], qWeight, qBias, m.embedDim); err != nil {
			return nil, err
		}
	}
	for token := 0; token < visionTokensCount; token++ {
		if err := packedResamplerProjection(k[token*m.embedDim:(token+1)*m.embedDim], keysInput[token*m.embedDim:(token+1)*m.embedDim], kWeight, kBias, m.embedDim); err != nil {
			return nil, err
		}
		if err := packedResamplerProjection(v[token*m.embedDim:(token+1)*m.embedDim], normKV[token*m.embedDim:(token+1)*m.embedDim], vWeight, vBias, m.embedDim); err != nil {
			return nil, err
		}
	}
	attention := make([]float32, m.numQuery*m.embedDim)
	scores := make([]float32, visionTokensCount)
	scale := float32(1 / math.Sqrt(float64(m.headDim)))
	for query := 0; query < m.numQuery; query++ {
		for head := 0; head < m.heads; head++ {
			qHead := q[query*m.embedDim+head*m.headDim : query*m.embedDim+(head+1)*m.headDim]
			for token := 0; token < visionTokensCount; token++ {
				kHead := k[token*m.embedDim+head*m.headDim : token*m.embedDim+(head+1)*m.headDim]
				scores[token] = simd.Sdot(qHead, kHead) * scale
			}
			if !simd.SoftmaxInPlace(scores) {
				return nil, fmt.Errorf("MiniCPM-V/O resampler attention softmax failed")
			}
			outHead := attention[query*m.embedDim+head*m.headDim : query*m.embedDim+(head+1)*m.headDim]
			for token := 0; token < visionTokensCount; token++ {
				vHead := v[token*m.embedDim+head*m.headDim : token*m.embedDim+(head+1)*m.headDim]
				simd.VecScaleAdd(outHead, outHead, vHead, scores[token])
			}
		}
	}
	projected := make([]float32, len(attention))
	for query := 0; query < m.numQuery; query++ {
		if err := m.outProj.forward(projected[query*m.embedDim:(query+1)*m.embedDim], attention[query*m.embedDim:(query+1)*m.embedDim]); err != nil {
			return nil, err
		}
	}
	normPost := make([]float32, len(projected))
	if !simd.LayerNormLastAxisTo(normPost, projected, m.numQuery, m.embedDim, m.lnPostWeight, m.lnPostBias, 1e-6) {
		return nil, fmt.Errorf("MiniCPM-V/O resampler post LayerNorm failed")
	}
	out := make([]float32, len(normPost))
	// Upstream uses x @ proj, whereas linear weights are applied as W @ x.
	for row := 0; row < m.numQuery; row++ {
		x := normPost[row*m.embedDim : (row+1)*m.embedDim]
		dst := out[row*m.embedDim : (row+1)*m.embedDim]
		for column := 0; column < m.embedDim; column++ {
			var sum float32
			for inner := 0; inner < m.embedDim; inner++ {
				sum += x[inner] * m.finalProj[inner*m.embedDim+column]
			}
			dst[column] = sum
		}
	}
	if err := validateFiniteTextOutput("resampler output", out); err != nil {
		return nil, err
	}
	return out, nil
}

func packedResamplerProjection(dst, input, weight, bias []float32, dim int) error {
	if len(dst) != dim || len(input) != dim || len(weight) != dim*dim || len(bias) != dim || !simd.GemvRows(dst, input, weight, dim, dim) {
		return fmt.Errorf("MiniCPM-V/O resampler attention projection failed")
	}
	for i := range dst {
		dst[i] += bias[i]
	}
	return nil
}

func resampler2DPositions(height, width, embedDim int) []float32 {
	if height <= 0 || width <= 0 || embedDim <= 0 || embedDim%4 != 0 {
		return nil
	}
	out := make([]float32, height*width*embedDim)
	axisDim := embedDim / 2
	frequencyDim := axisDim / 2
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			row := out[(y*width+x)*embedDim : (y*width+x+1)*embedDim]
			// Upstream np.meshgrid(grid_w, grid_h) feeds grid[0] first.
			for axis, position := range []int{x, y} {
				base := axis * axisDim
				for i := 0; i < frequencyDim; i++ {
					omega := 1 / math.Pow(10000, float64(i)/float64(frequencyDim))
					angle := float64(position) * omega
					row[base+i] = float32(math.Sin(angle))
					row[base+frequencyDim+i] = float32(math.Cos(angle))
				}
			}
		}
	}
	return out
}

var _ Resampler = (*ResamplerCPU)(nil)
