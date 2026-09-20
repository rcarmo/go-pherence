package modernbert

import (
	"encoding/json"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type Config struct {
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	Layers           int      `json:"num_hidden_layers"`
	Heads            int      `json:"num_attention_heads"`
	MaxPositions     int      `json:"max_position_embeddings"`
	GlobalEvery      int      `json:"global_attn_every_n_layers"`
	LocalAttention   int      `json:"local_attention"`
	NormEps          float64  `json:"norm_eps"`
	PadID            int      `json:"pad_token_id"`
	HiddenActivation string   `json:"hidden_activation"`
	AttentionBias    bool     `json:"attention_bias"`
	MLPBias          bool     `json:"mlp_bias"`
	NormBias         bool     `json:"norm_bias"`
	LayerTypes       []string `json:"layer_types"`
	RopeParameters   map[string]struct {
		Theta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}
type layerWeights struct{ attnNorm, qkv, attnOut, mlpNorm, mlpIn, mlpOut []float32 }
type modelWeights struct {
	embedding, embeddingNorm, finalNorm []float32
	layers                              []layerWeights
}
type Tensor struct {
	Shape []int
	Data  []float32
}
type Model struct {
	Config  Config
	tensors map[string]Tensor
	weights modelWeights
}

func ParseConfig(raw json.RawMessage) (Config, error) {
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, err
	}
	if c.VocabSize < 4 || c.VocabSize > 1<<20 || c.HiddenSize < 2 || c.HiddenSize > 16384 || c.IntermediateSize < 1 || c.IntermediateSize > 65536 || c.Layers < 1 || c.Layers > 256 || c.Heads < 1 || c.Heads > 256 || c.HiddenSize%c.Heads != 0 || c.MaxPositions < 1 || c.MaxPositions > 1<<20 || c.LocalAttention < 2 || c.LocalAttention > c.MaxPositions*2 || c.NormEps <= 0 || c.NormEps > 1 || c.PadID < 0 || c.PadID >= c.VocabSize {
		return c, fmt.Errorf("modernbert: invalid config")
	}
	if c.HiddenActivation != "gelu" || c.AttentionBias || c.MLPBias || c.NormBias {
		return c, fmt.Errorf("modernbert: only bias-free GELU checkpoints supported")
	}
	if c.RopeParameters == nil {
		c.RopeParameters = map[string]struct {
			Theta float64 `json:"rope_theta"`
		}{"full_attention": {Theta: 160000}, "sliding_attention": {Theta: 10000}}
	}
	if len(c.LayerTypes) == 0 {
		for i := 0; i < c.Layers; i++ {
			kind := "sliding_attention"
			if c.GlobalEvery > 0 && i%c.GlobalEvery == 0 {
				kind = "full_attention"
			}
			c.LayerTypes = append(c.LayerTypes, kind)
		}
	}
	if len(c.LayerTypes) != c.Layers {
		return c, fmt.Errorf("modernbert: layer type count")
	}
	for _, kind := range c.LayerTypes {
		if kind != "full_attention" && kind != "sliding_attention" {
			return c, fmt.Errorf("modernbert: unsupported layer type %q", kind)
		}
	}
	for _, kind := range []string{"full_attention", "sliding_attention"} {
		if c.RopeParameters[kind].Theta <= 0 {
			return c, fmt.Errorf("modernbert: invalid %s rope", kind)
		}
	}
	return c, nil
}
func New(raw json.RawMessage, tensors map[string]Tensor) (*Model, error) {
	return newModel(raw, tensors, false)
}
func newModel(raw json.RawMessage, tensors map[string]Tensor, takeOwnership bool) (*Model, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	shape := func(name string, want ...int) error {
		v, ok := tensors[name]
		if !ok {
			return fmt.Errorf("modernbert: missing %s", name)
		}
		if len(v.Shape) != len(want) {
			return fmt.Errorf("modernbert: %s rank", name)
		}
		n := 1
		for i, d := range want {
			if d < 1 || v.Shape[i] != d {
				return fmt.Errorf("modernbert: %s shape %v want %v", name, v.Shape, want)
			}
			n *= d
		}
		if len(v.Data) != n {
			return fmt.Errorf("modernbert: %s length", name)
		}
		for _, x := range v.Data {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return fmt.Errorf("modernbert: %s nonfinite", name)
			}
		}
		return nil
	}
	h, inter := c.HiddenSize, c.IntermediateSize
	expected := make(map[string]struct{}, 6*c.Layers+2)
	need := func(name string, dims ...int) error { expected[name] = struct{}{}; return shape(name, dims...) }
	if err := need("embeddings.tok_embeddings.weight", c.VocabSize, h); err != nil {
		return nil, err
	}
	if err := need("embeddings.norm.weight", h); err != nil {
		return nil, err
	}
	if err := need("final_norm.weight", h); err != nil {
		return nil, err
	}
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		if l > 0 {
			if err := need(p+"attn_norm.weight", h); err != nil {
				return nil, err
			}
		}
		for name, want := range map[string][]int{"attn.Wqkv.weight": {3 * h, h}, "attn.Wo.weight": {h, h}, "mlp_norm.weight": {h}, "mlp.Wi.weight": {2 * inter, h}, "mlp.Wo.weight": {h, inter}} {
			if err := need(p+name, want...); err != nil {
				return nil, err
			}
		}
	}
	for name := range tensors {
		if _, ok := expected[name]; !ok {
			return nil, fmt.Errorf("modernbert: unexpected tensor %s", name)
		}
	}
	own := make(map[string]Tensor, len(tensors))
	for k, v := range tensors {
		if takeOwnership {
			own[k] = v
		} else {
			own[k] = Tensor{Shape: append([]int(nil), v.Shape...), Data: append([]float32(nil), v.Data...)}
		}
	}
	m := &Model{Config: c, tensors: own}
	m.weights.embedding = own["embeddings.tok_embeddings.weight"].Data
	m.weights.embeddingNorm = own["embeddings.norm.weight"].Data
	m.weights.finalNorm = own["final_norm.weight"].Data
	m.weights.layers = make([]layerWeights, c.Layers)
	for l := range m.weights.layers {
		p := fmt.Sprintf("layers.%d.", l)
		w := &m.weights.layers[l]
		if l > 0 {
			w.attnNorm = own[p+"attn_norm.weight"].Data
		}
		w.qkv = own[p+"attn.Wqkv.weight"].Data
		w.attnOut = own[p+"attn.Wo.weight"].Data
		w.mlpNorm = own[p+"mlp_norm.weight"].Data
		w.mlpIn = own[p+"mlp.Wi.weight"].Data
		w.mlpOut = own[p+"mlp.Wo.weight"].Data
	}
	return m, nil
}
func (m *Model) tensor(name string) []float32 { return m.tensors[name].Data }
func layerNorm(dst, src, w []float32, rows, cols int, eps float32) {
	for r := 0; r < rows; r++ {
		x := src[r*cols : (r+1)*cols]
		var mean float32
		for _, v := range x {
			mean += v
		}
		mean /= float32(cols)
		var ss float32
		for _, v := range x {
			d := v - mean
			ss += d * d
		}
		inv := float32(1 / math.Sqrt(float64(ss/float32(cols)+eps)))
		for j, v := range x {
			dst[r*cols+j] = (v - mean) * inv * w[j]
		}
	}
}
func rope(rows []float32, seq, heads, dim int, theta float64) {
	half := dim / 2
	for pos := 0; pos < seq; pos++ {
		for h := 0; h < heads; h++ {
			base := (pos*heads + h) * dim
			for j := 0; j < half; j++ {
				angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(dim))
				co, si := float32(math.Cos(angle)), float32(math.Sin(angle))
				a, b := rows[base+j], rows[base+j+half]
				rows[base+j] = a*co - b*si
				rows[base+j+half] = b*co + a*si
			}
		}
	}
}
func (m *Model) Forward(ids []int, mask []bool) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("modernbert: nil model")
	}
	s, err := m.NewSession(len(ids))
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(ids)*m.Config.HiddenSize)
	if err = s.ForwardInto(out, ids, mask); err != nil {
		return nil, err
	}
	return out, nil
}
func (m *Model) ForwardLayers(ids []int, mask []bool) ([]float32, [][]float32, error) {
	c := m.Config
	n, h, inter := len(ids), c.HiddenSize, c.IntermediateSize
	if n < 1 || n > c.MaxPositions || len(mask) != n {
		return nil, nil, fmt.Errorf("modernbert: invalid input")
	}
	hidden := make([]float32, n*h)
	emb := m.tensor("embeddings.tok_embeddings.weight")
	for i, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, nil, fmt.Errorf("modernbert: token")
		}
		copy(hidden[i*h:], emb[id*h:(id+1)*h])
	}
	normed := make([]float32, len(hidden))
	layerNorm(normed, hidden, m.tensor("embeddings.norm.weight"), n, h, float32(c.NormEps))
	hidden = normed
	states := [][]float32{append([]float32(nil), hidden...)}
	hd := h / c.Heads
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("layers.%d.", l)
		input := hidden
		if l > 0 {
			input = make([]float32, len(hidden))
			layerNorm(input, hidden, m.tensor(p+"attn_norm.weight"), n, h, float32(c.NormEps))
		}
		qkv := make([]float32, n*3*h)
		simd.DenseNTTo(qkv, input, m.tensor(p+"attn.Wqkv.weight"), n, 3*h, h, 1, h, h, 3*h)
		q, k, v := make([]float32, n*h), make([]float32, n*h), make([]float32, n*h)
		for i := 0; i < n; i++ {
			copy(q[i*h:], qkv[i*3*h:i*3*h+h])
			copy(k[i*h:], qkv[i*3*h+h:i*3*h+2*h])
			copy(v[i*h:], qkv[i*3*h+2*h:i*3*h+3*h])
		}
		theta := 10000.
		if c.LayerTypes[l] == "full_attention" {
			theta = 160000
		}
		if rp, ok := c.RopeParameters[c.LayerTypes[l]]; ok && rp.Theta > 0 {
			theta = rp.Theta
		}
		rope(q, n, c.Heads, hd, theta)
		rope(k, n, c.Heads, hd, theta)
		attn := make([]float32, n*h)
		scores := make([]float32, n*n)
		scale := float32(1 / math.Sqrt(float64(hd)))
		window := c.LocalAttention / 2
		for head := 0; head < c.Heads; head++ {
			for i := 0; i < n; i++ {
				row := scores[i*n : (i+1)*n]
				for j := 0; j < n; j++ {
					allowed := mask[j]
					if c.LayerTypes[l] == "sliding_attention" && (j < i-window || j > i+window) {
						allowed = false
					}
					if !allowed {
						row[j] = float32(math.Inf(-1))
						continue
					}
					var sum float32
					for d := 0; d < hd; d++ {
						sum += q[i*h+head*hd+d] * k[j*h+head*hd+d]
					}
					row[j] = sum * scale
				}
				simd.SoftmaxRowsInPlace(row, 1, n)
				for d := 0; d < hd; d++ {
					var sum float32
					for j, w := range row {
						sum += w * v[j*h+head*hd+d]
					}
					attn[i*h+head*hd+d] = sum
				}
			}
		}
		proj := make([]float32, n*h)
		simd.DenseNTTo(proj, attn, m.tensor(p+"attn.Wo.weight"), n, h, h, 1, h, h, h)
		for i := range hidden {
			hidden[i] += proj[i]
		}
		mlpin := make([]float32, len(hidden))
		layerNorm(mlpin, hidden, m.tensor(p+"mlp_norm.weight"), n, h, float32(c.NormEps))
		wi := make([]float32, n*2*inter)
		simd.DenseNTTo(wi, mlpin, m.tensor(p+"mlp.Wi.weight"), n, 2*inter, h, 1, h, h, 2*inter)
		mid := make([]float32, n*inter)
		for i := range mid {
			a, b := wi[(i/inter)*2*inter+i%inter], wi[(i/inter)*2*inter+inter+i%inter]
			mid[i] = float32(.5*float64(a)*(1+math.Erf(float64(a)*.7071067811865476))) * b
		}
		mo := make([]float32, n*h)
		simd.DenseNTTo(mo, mid, m.tensor(p+"mlp.Wo.weight"), n, h, inter, 1, inter, inter, h)
		for i := range hidden {
			hidden[i] += mo[i]
		}
		states = append(states, append([]float32(nil), hidden...))
	}
	final := make([]float32, len(hidden))
	layerNorm(final, hidden, m.tensor("final_norm.weight"), n, h, float32(c.NormEps))
	return final, states, nil
}
