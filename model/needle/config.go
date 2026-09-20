// Package needle implements the distinct Needle 2 and Needle 3 architectures.
// The CPU reference path uses FP32 and native SIMD for dense forward/backward work.
package needle

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type Config struct {
	Generation        int     `json:"generation,omitempty"`
	ArchiveDecoded    bool    `json:"go_archive_decoded,omitempty"`
	ArchiveKVWindow   int     `json:"go_archive_kv_window,omitempty"`
	VocabSize         int     `json:"vocab_size"`
	DModel            int     `json:"d_model"`
	AttnDim           int     `json:"attn_dim,omitempty"`
	Heads             int     `json:"num_heads"`
	KVHeads           int     `json:"num_kv_heads"`
	Layers            int     `json:"num_layers"`
	QKDim             int     `json:"qk_head_dim"`
	VDim              int     `json:"v_head_dim"`
	MaxSeq            int     `json:"max_seq_len"`
	RopeTheta         float64 `json:"rope_theta"`
	Lanes             int     `json:"mhc_lanes"`
	ConvTaps          int     `json:"qkv_conv_taps"`
	Window            int     `json:"sliding_window"`
	GlobalLayers      []int   `json:"global_layers"`
	EngramLayers      []int   `json:"engram_layers"`
	EngramOrders      []int   `json:"engram_orders"`
	EngramHeads       int     `json:"engram_heads"`
	EngramSlots       int     `json:"engram_slots"`
	EngramSeedHeads   int     `json:"engram_seed_heads"`
	LadderWidths      []int   `json:"ladder_widths"`
	LadderOrder       []int   `json:"ladder_order"`
	OutVocab          int     `json:"out_vocab"`
	DType             string  `json:"dtype"`
	PadID             int     `json:"pad_token_id"`
	EmbeddingDim      int     `json:"embedding_dim"`
	ContrastiveDim    int     `json:"contrastive_dim"`
	EmbeddingProbes   int     `json:"embedding_probes"`
	EmbeddingQueries  int     `json:"embedding_queries"`
	ConfidenceProbes  int     `json:"confidence_probes"`
	ConfidenceQueries int     `json:"confidence_queries"`
	RouterProbes      int     `json:"router_probes"`
	RouterQueries     int     `json:"router_queries"`
}

func (c *Config) validate() error {
	if c.Generation != 2 && c.Generation != 3 {
		return fmt.Errorf("needle: generation must be 2 or 3")
	}
	if c.ArchiveDecoded && (c.ArchiveKVWindow < 0 || c.ArchiveKVWindow > c.MaxSeq) {
		return fmt.Errorf("needle: invalid decoded archive metadata")
	}
	if c.Generation == 2 {
		if c.AttnDim == 0 {
			c.AttnDim = c.DModel
		}
		if c.Heads > 0 {
			c.QKDim = c.AttnDim / c.Heads
			c.VDim = c.QKDim
		}
		c.ConvTaps = 0
		c.Window = 0
	}
	if c.QKDim == 0 && c.Heads > 0 {
		c.QKDim = c.DModel / c.Heads
	}
	if c.VDim == 0 && c.Heads > 0 {
		c.VDim = c.DModel / c.Heads
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 100000
	}
	if c.Lanes == 0 {
		c.Lanes = 4
	}
	if c.OutVocab == 0 {
		c.OutVocab = c.VocabSize
	}
	if c.DModel < 2 || c.DModel > 4096 || c.VocabSize < 4 || c.VocabSize > 131072 || c.OutVocab < 1 || c.OutVocab > c.VocabSize || c.Heads < 1 || c.Heads > 128 || c.KVHeads < 1 || c.KVHeads > c.Heads || c.Heads%c.KVHeads != 0 || c.Layers < 1 || c.Layers > 128 || c.QKDim < 2 || c.QKDim > 512 || c.QKDim%2 != 0 || c.VDim < 1 || c.VDim > 512 || c.MaxSeq < 1 || c.MaxSeq > 65536 || c.Lanes < 1 || c.Lanes > 8 || c.ConvTaps < 0 || c.ConvTaps > 32 || c.Window < 0 || c.Window > c.MaxSeq || c.RopeTheta <= 0 || math.IsNaN(c.RopeTheta) || math.IsInf(c.RopeTheta, 0) {
		return fmt.Errorf("needle: invalid or excessive model geometry")
	}
	if c.PadID < 0 || c.PadID >= c.VocabSize {
		return fmt.Errorf("needle: invalid padding token")
	}
	if c.EmbeddingDim == 0 {
		c.EmbeddingDim = 128
	}
	if c.EmbeddingDim < 1 || c.EmbeddingDim > 4096 {
		return fmt.Errorf("needle: invalid embedding dimension")
	}
	for _, n := range []*int{&c.EmbeddingProbes, &c.EmbeddingQueries, &c.ConfidenceProbes, &c.ConfidenceQueries, &c.RouterProbes, &c.RouterQueries} {
		if *n == 0 {
			*n = 4
		}
		if *n < 1 || *n > 64 {
			return fmt.Errorf("needle: invalid head geometry")
		}
	}
	if c.Generation == 2 && c.AttnDim%c.Heads != 0 {
		return fmt.Errorf("needle: attn_dim must divide heads")
	}
	for _, ls := range [][]int{c.EngramLayers, c.GlobalLayers} {
		seen := map[int]bool{}
		for _, l := range ls {
			if l < 0 || l >= c.Layers || seen[l] {
				return fmt.Errorf("needle: invalid/duplicate layer index")
			}
			seen[l] = true
		}
	}
	if !slices.IsSorted(c.EngramLayers) {
		return fmt.Errorf("needle: engram_layers must be sorted")
	}
	if len(c.EngramLayers) > 0 {
		if len(c.EngramOrders) == 0 {
			c.EngramOrders = []int{2, 3}
		}
		if len(c.EngramOrders) > 8 {
			return fmt.Errorf("needle: too many engram orders")
		}
		for _, o := range c.EngramOrders {
			if o < 1 || o > 32 {
				return fmt.Errorf("needle: invalid engram order")
			}
		}
		if c.EngramHeads == 0 {
			c.EngramHeads = max(1, c.DModel/(len(c.EngramOrders)*128))
		}
		if c.EngramHeads < 1 || c.EngramHeads > 128 || c.EngramSlots < 1 || c.EngramSlots > 1048576 || c.DModel%(len(c.EngramOrders)*c.EngramHeads) != 0 || c.EngramSeedHeads < 0 || c.EngramSeedHeads > 128 {
			return fmt.Errorf("needle: invalid engram geometry")
		}
	}
	if len(c.LadderWidths) > 0 && (c.DModel&(c.DModel-1) != 0) {
		return fmt.Errorf("needle: split Hadamard permutations require power-of-two width")
	}
	return nil
}

// Model owns an immutable copy of a checkpoint. Training returns a new checkpoint.
// Forward and LossGrad may run concurrently; caller-owned tensors never alias it.
type Model struct {
	config        Config
	tensors       map[string]checkpoint.Tensor
	rawConfig     json.RawMessage
	p1, p2        []int
	deployed      bool
	archiveWindow int
	packed        map[packedKey]*simd.CQMatrix // archive hybrid path, immutable
}

func Load(path string) (*Model, error) {
	c, e := checkpoint.Load(path)
	if e != nil {
		return nil, e
	}
	return New(c)
}
func New(cp *checkpoint.Checkpoint) (*Model, error) { return newModel(cp, false) }

// takeOwnership is restricted to freshly built, private archive checkpoints.
// Public New always copies caller-owned tensors.
func newModel(cp *checkpoint.Checkpoint, takeOwnership bool) (*Model, error) {
	if cp == nil {
		return nil, fmt.Errorf("needle: nil checkpoint")
	}
	if cp.FormatVersion != 2 {
		return nil, fmt.Errorf("needle: checkpoint format %d unsupported; expected 2", cp.FormatVersion)
	}
	var c Config
	if e := json.Unmarshal(cp.Config, &c); e != nil {
		return nil, e
	}
	if c.Generation == 0 {
		if c.QKDim > 0 {
			c.Generation = 3
		} else if c.AttnDim > 0 {
			c.Generation = 2
		} else {
			return nil, fmt.Errorf("needle: ambiguous generation; specify generation in checkpoint config")
		}
	}
	if e := c.validate(); e != nil {
		return nil, e
	}
	m := &Model{config: c, tensors: make(map[string]checkpoint.Tensor), rawConfig: append(json.RawMessage(nil), cp.Config...), deployed: c.ArchiveDecoded, archiveWindow: c.ArchiveKVWindow}
	if err := validateAB(cp.Tensors); err != nil {
		return nil, err
	}
	shapes := expectedShapes(c)
	for name, want := range shapes {
		got, ok := cp.Tensors[name]
		if !ok || !slices.Equal(got.Shape, want) {
			return nil, fmt.Errorf("needle: missing/invalid tensor %s: want %v", name, want)
		}
	}
	var total int64
	// Preserve auxiliary heads and calibration tensors on checkpoint round-trip;
	// the trunk loss does not train them, but must not destroy them on save.
	for name, got := range cp.Tensors {
		want := got.Shape
		if len(want) > 8 {
			return nil, fmt.Errorf("needle: excessive tensor rank %s", name)
		}
		n := 1
		for _, d := range want {
			if d <= 0 || n > (1<<29)/d {
				return nil, fmt.Errorf("needle: excessive tensor %s", name)
			}
			n *= d
		}
		total += int64(n)
		if total > 1<<29 || len(got.Data) != n {
			return nil, fmt.Errorf("needle: invalid/excessive tensor storage %s", name)
		}
		for _, v := range got.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("needle: nonfinite tensor %s", name)
			}
		}
		data := got.Data
		if !takeOwnership {
			data = append([]float32(nil), data...)
		}
		m.tensors[name] = checkpoint.Tensor{Shape: append([]int{}, got.Shape...), Data: data}
	}
	n := padded(c.DModel)
	m.p1 = numpyPermutation(n, 11, len(c.LadderWidths) > 0)
	m.p2 = numpyPermutation(n, 13, len(c.LadderWidths) > 0)
	return m, nil
}
func padded(d int) int {
	n := 1
	for n < d {
		n *= 2
	}
	return n
}
func hadaBlocks(n int) (int, int) {
	a := 1
	for a*a*4 <= n {
		a *= 2
	}
	return a, n / a
}
func expectedShapes(c Config) map[string][]int {
	d, l, h, q, v, n := c.DModel, c.Layers, c.Lanes, c.QKDim, c.VDim, padded(c.DModel)
	s := map[string][]int{"embedding/embedding": {c.VocabSize, d}, "stack/final_norm/scale": {d}}
	for _, k := range []string{"pre", "post", "res"} {
		cols := h
		if k == "res" {
			cols = h * h
		}
		s["stack/mhc_phi_"+k] = []int{l, h * d, cols}
		s["stack/mhc_a_"+k] = []int{l}
		if k == "res" {
			s["stack/mhc_b_"+k] = []int{l, h, h}
		} else {
			s["stack/mhc_b_"+k] = []int{l, h}
		}
	}
	b := "stack/layers/block/"
	for _, k := range []string{"ZCRMSNorm_0", "post_attn_norm", "pre_hada_norm"} {
		s[b+k+"/scale"] = []int{l, d}
	}
	s[b+"attn_gate"] = []int{l}
	a := b + "self_attn/"
	for key, cols := range map[string]int{"q_proj": c.Heads * q, "k_proj": c.KVHeads * q, "v_proj": c.KVHeads * v, "gate_proj": c.Heads * v} {
		s[a+key+"/kernel"] = []int{l, d, cols}
	}
	s[a+"out_proj/kernel"] = []int{l, c.Heads * v, d}
	s[a+"q_norm/scale"] = []int{l, q}
	s[a+"k_norm/scale"] = []int{l, q}
	if c.ConvTaps > 0 {
		for key, cols := range map[string]int{"q_taps": c.Heads * q, "k_taps": c.KVHeads * q, "v_taps": c.KVHeads * v} {
			s[a+key] = []int{l, c.ConvTaps, cols}
		}
	}
	hm := b + "hadamard_mlp/"
	for _, key := range []string{"d1", "d2", "d3"} {
		s[hm+key] = []int{l, n}
	}
	if c.Generation == 3 {
		ba, bb := hadaBlocks(n)
		for j := 1; j <= 3; j++ {
			s[fmt.Sprintf("%sw%da", hm, j)] = []int{l, ba, ba}
			s[fmt.Sprintf("%sw%db", hm, j)] = []int{l, bb, bb}
		}
		for _, key := range []string{"d4", "b2"} {
			s[hm+key] = []int{l, n}
		}
		s[hm+"cond_v"] = []int{l, d, 8}
		s[hm+"cond_u"] = []int{l, 8, n}
	}
	for i := range c.EngramLayers {
		p := fmt.Sprintf("engrams_%d/", i)
		tables := len(c.EngramOrders) * c.EngramHeads
		s[p+"embedding"] = []int{tables, c.EngramSlots, d / tables}
		s[p+"key_proj/kernel"] = []int{d, d}
		s[p+"value_proj/kernel"] = []int{d, d}
		s[p+"taps"] = []int{4, d}
	}
	return s
}
