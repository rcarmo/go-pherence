package needle

import (
	"encoding/json"
	"fmt"
	"slices"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// LoadArchive materializes a Needle3 deployment archive as FP32 tensors once.
// This is compatible archive inference, not a packed SIMD kernel or KV cache.
// Currently only KV8 archives are executable; narrower KV formats fail closed.
func LoadArchive(path string) (*Model, *checkpoint.Tokenizer, error) {
	a, err := checkpoint.LoadArchive(path)
	if err != nil {
		return nil, nil, err
	}
	return modelFromArchive(a)
}
func modelFromArchive(a *checkpoint.Archive) (*Model, *checkpoint.Tokenizer, error) {
	if a == nil {
		return nil, nil, fmt.Errorf("needle: nil archive")
	}
	h := a.Header
	if h[4] != 8 {
		return nil, nil, fmt.Errorf("needle: archive KV%d execution not implemented", h[4])
	}
	c := Config{Generation: 3, VocabSize: int(h[5]), OutVocab: int(h[6]), DModel: int(h[7]), Heads: int(h[8]), KVHeads: int(h[9]), Layers: int(h[10]), QKDim: int(h[11]), VDim: int(h[12]), MaxSeq: int(h[13]), Lanes: int(h[15]), Window: int(h[16]), ConvTaps: int(h[19]), EngramSlots: int(h[20]), EngramSeedHeads: int(h[25]), RopeTheta: float64(a.RopeTheta), DType: "float32"}
	for i := 0; i < int(h[26]); i++ {
		c.EngramOrders = append(c.EngramOrders, int(h[27+i]))
	}
	if len(c.EngramOrders) > 0 {
		c.EngramHeads = int(h[22]) / len(c.EngramOrders)
	}
	for i := 0; i < int(h[31]); i++ {
		c.EngramLayers = append(c.EngramLayers, int(h[32+i]))
	}
	gmask := uint64(h[17]) | uint64(h[18])<<32
	for i := 0; i < c.Layers; i++ {
		if gmask&(1<<i) != 0 {
			c.GlobalLayers = append(c.GlobalLayers, i)
		}
	}
	if err := c.validate(); err != nil {
		return nil, nil, err
	}
	// Check expanded stacked shapes before materializing even the first layer.
	var expanded int64
	for _, shape := range expectedShapes(c) {
		n := int64(1)
		for _, d := range shape {
			if int64(d) > (512<<20)/4/n {
				return nil, nil, fmt.Errorf("needle: expanded archive exceeds 512 MiB")
			}
			n *= int64(d)
		}
		expanded += n * 4
		if expanded > 512<<20 {
			return nil, nil, fmt.Errorf("needle: expanded archive exceeds 512 MiB")
		}
	}
	// Account all source records (including optional heads), output restacking,
	// transposes, final owned copies and later destruction conservatively. This
	// is a logical admission limit, not a process RSS promise. A caller can
	// still hold unrelated models; no hidden unbounded copy plan is admitted.
	var decoded int64
	for _, r := range a.Records {
		decoded += int64(len(r.Data))*4 + int64(len(r.Raw))
	}
	if decoded > (1<<30)/4 || expanded > (1<<30)/3 || decoded*4+expanded*3 > 1<<30 {
		return nil, nil, fmt.Errorf("needle: archive materialization exceeds 1 GiB logical peak budget")
	}
	if int(h[14]) != padded(c.DModel) || h[23] != 4 || int(h[24]) != slices.Max(c.EngramOrders) {
		return nil, nil, fmt.Errorf("needle: unsupported archive Hadamard/engram geometry")
	}
	cp := &checkpoint.Checkpoint{FormatVersion: 2, Tensors: map[string]checkpoint.Tensor{}}
	cursor := 0
	take := func(shape []int) (checkpoint.Tensor, error) {
		if cursor >= len(a.Records) {
			return checkpoint.Tensor{}, fmt.Errorf("needle: missing archive tensor %d", cursor)
		}
		r := a.Records[cursor]
		cursor++
		if r.DType == 4 || !slices.Equal(r.Shape, shape) {
			return checkpoint.Tensor{}, fmt.Errorf("needle: archive tensor %d shape %v want %v", cursor-1, r.Shape, shape)
		}
		n := 1
		for _, s := range shape {
			n *= s
		}
		if n != len(r.Data) {
			return checkpoint.Tensor{}, fmt.Errorf("needle: archive tensor length mismatch")
		}
		return checkpoint.Tensor{Shape: append([]int{}, shape...), Data: r.Data}, nil
	}
	transpose := func(w checkpoint.Tensor) checkpoint.Tensor {
		rows, cols := w.Shape[0], w.Shape[1]
		out := checkpoint.Tensor{Shape: []int{cols, rows}, Data: make([]float32, len(w.Data))}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out.Data[j*rows+i] = w.Data[i*cols+j]
			}
		}
		return out
	}
	put := func(name string, shape []int, trans bool, layer int) error {
		record, err := take(shape)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if trans {
			record = transpose(record)
		}
		if layer >= 0 {
			w, ok := cp.Tensors[name]
			if !ok {
				w = checkpoint.Tensor{Shape: append([]int{c.Layers}, record.Shape...), Data: make([]float32, c.Layers*len(record.Data))}
			}
			copy(w.Data[layer*len(record.Data):], record.Data)
			cp.Tensors[name] = w
		} else {
			cp.Tensors[name] = record
		}
		return nil
	}
	d, l, n := c.DModel, c.Layers, padded(c.DModel)
	if err := put("embedding/embedding", []int{c.VocabSize, d}, false, -1); err != nil {
		return nil, nil, err
	}
	ba, bb := hadaBlocks(n)
	type field struct {
		name  string
		shape []int
		trans bool
	}
	for layer := 0; layer < l; layer++ {
		fields := []field{{"ZCRMSNorm_0/scale", []int{d}, false}, {"self_attn/q_proj/kernel", []int{c.Heads * c.QKDim, d}, true}, {"self_attn/k_proj/kernel", []int{c.KVHeads * c.QKDim, d}, true}, {"self_attn/v_proj/kernel", []int{c.KVHeads * c.VDim, d}, true}}
		if c.ConvTaps > 0 {
			fields = append(fields, field{"self_attn/q_taps", []int{c.ConvTaps, c.Heads * c.QKDim}, false}, field{"self_attn/k_taps", []int{c.ConvTaps, c.KVHeads * c.QKDim}, false}, field{"self_attn/v_taps", []int{c.ConvTaps, c.KVHeads * c.VDim}, false})
		}
		fields = append(fields, field{"self_attn/q_norm/scale", []int{c.QKDim}, false}, field{"self_attn/k_norm/scale", []int{c.QKDim}, false}, field{"self_attn/gate_proj/kernel", []int{c.Heads * c.VDim, d}, true}, field{"self_attn/out_proj/kernel", []int{d, c.Heads * c.VDim}, true}, field{"post_attn_norm/scale", []int{d}, false}, field{"attn_gate", []int{1}, false}, field{"pre_hada_norm/scale", []int{d}, false})
		for _, name := range []string{"d1", "d2", "b2", "d3", "d4"} {
			fields = append(fields, field{"hadamard_mlp/" + name, []int{n}, false})
		}
		for i := 1; i <= 3; i++ {
			fields = append(fields, field{fmt.Sprintf("hadamard_mlp/w%da", i), []int{ba, ba}, false}, field{fmt.Sprintf("hadamard_mlp/w%db", i), []int{bb, bb}, false})
		}
		fields = append(fields, field{"hadamard_mlp/cond_v", []int{d, 8}, false}, field{"hadamard_mlp/cond_u", []int{8, n}, false})
		for _, f := range fields {
			if err := put("stack/layers/block/"+f.name, f.shape, f.trans, layer); err != nil {
				return nil, nil, err
			}
		}
	}
	gate := cp.Tensors["stack/layers/block/attn_gate"]
	gate.Shape = []int{l}
	cp.Tensors["stack/layers/block/attn_gate"] = gate
	for _, kind := range []string{"a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res"} {
		shape := []int{l}
		if kind[0] == 'b' {
			shape = append(shape, c.Lanes)
			if kind == "b_res" {
				shape = append(shape, c.Lanes)
			}
		}
		if err := put("stack/mhc_"+kind, shape, false, -1); err != nil {
			return nil, nil, err
		}
	}
	for _, kind := range []string{"pre", "post", "res"} {
		cols := c.Lanes
		if kind == "res" {
			cols *= c.Lanes
		}
		r, err := take([]int{l * cols, c.Lanes * d})
		if err != nil {
			return nil, nil, err
		}
		out := checkpoint.Tensor{Shape: []int{l, c.Lanes * d, cols}, Data: make([]float32, len(r.Data))}
		for layer := 0; layer < l; layer++ {
			for i := 0; i < cols; i++ {
				for j := 0; j < c.Lanes*d; j++ {
					out.Data[(layer*c.Lanes*d+j)*cols+i] = r.Data[(layer*cols+i)*c.Lanes*d+j]
				}
			}
		}
		cp.Tensors["stack/mhc_phi_"+kind] = out
	}
	perms := make([][]int, 2)
	for i := range perms {
		r, err := take([]int{n})
		if err != nil {
			return nil, nil, err
		}
		perms[i] = make([]int, n)
		seen := make([]bool, n)
		for j, v := range r.Data {
			id := int(v)
			if float32(id) != v || id < 0 || id >= n || seen[id] {
				return nil, nil, fmt.Errorf("needle: invalid archive permutation")
			}
			seen[id] = true
			perms[i][j] = id
		}
	}
	for site := range c.EngramLayers {
		prefix := fmt.Sprintf("engrams_%d/", site)
		tables, sub := int(h[22]), int(h[21])
		r, err := take([]int{tables * c.EngramSlots, sub})
		if err != nil {
			return nil, nil, err
		}
		r.Shape = []int{tables, c.EngramSlots, sub}
		cp.Tensors[prefix+"embedding"] = r
		for _, key := range []string{"key_proj/kernel", "value_proj/kernel"} {
			if err = put(prefix+key, []int{d, d}, true, -1); err != nil {
				return nil, nil, err
			}
		}
		if err = put(prefix+"taps", []int{4, d}, false, -1); err != nil {
			return nil, nil, err
		}
	}
	if err := put("stack/final_norm/scale", []int{d}, false, -1); err != nil {
		return nil, nil, err
	}
	if cursor < len(a.Records) && a.Records[cursor].DType != 4 {
		manifest := a.Records[cursor]
		if len(manifest.Shape) != 1 || manifest.Shape[0] < 1 || manifest.Shape[0] > 3 || len(manifest.Data) != manifest.Shape[0] {
			return nil, nil, fmt.Errorf("needle: invalid head manifest")
		}
		cursor++
		last := 0
		for _, code := range manifest.Data {
			id := int(code)
			if float32(id) != code || id <= last || id > 3 {
				return nil, nil, fmt.Errorf("needle: invalid head manifest code")
			}
			last = id
			kind := []string{"", "embedding", "confidence", "router"}[id]
			prefix := kind + "_head/"
			if cursor+5 >= len(a.Records) {
				return nil, nil, fmt.Errorf("needle: missing head tensors")
			}
			gain := a.Records[cursor+1]
			query := a.Records[cursor+2]
			proj := a.Records[cursor+4]
			if len(gain.Shape) != 2 || gain.Shape[0] != l+1 || gain.Shape[1] < 1 || gain.Shape[1] > 64 || len(query.Shape) != 2 || query.Shape[1] != d || query.Shape[0] < 1 || query.Shape[0] > 64 || len(proj.Shape) != 2 || proj.Shape[0] < 1 || proj.Shape[0] > 4096 || proj.Shape[1] != query.Shape[0]*d {
				return nil, nil, fmt.Errorf("needle: invalid archive head geometry")
			}
			k, q, out := gain.Shape[1], query.Shape[0], proj.Shape[0]
			shape := []int{(l + 1) * k, d}
			if len(a.Records[cursor].Shape) == 3 {
				shape = []int{l + 1, k, d}
			}
			r, err := take(shape)
			if err != nil {
				return nil, nil, err
			}
			r.Shape = []int{l + 1, k, d}
			cp.Tensors[prefix+"probes"] = r
			for _, f := range []field{{"gain", []int{l + 1, k}, false}, {"query", []int{q, d}, false}, {"row_bias", []int{q, l + 1, k}, false}, {"proj/kernel", []int{out, q * d}, true}, {"proj/bias", []int{out}, false}} {
				if err := put(prefix+f.name, f.shape, f.trans, -1); err != nil {
					return nil, nil, err
				}
			}
			switch id {
			case 1:
				c.EmbeddingDim, c.EmbeddingProbes, c.EmbeddingQueries = out, k, q
			case 2:
				c.ConfidenceProbes, c.ConfidenceQueries = k, q
			case 3:
				c.RouterProbes, c.RouterQueries = k, q
				if err := put(prefix+"calibration", []int{3}, false, -1); err != nil {
					return nil, nil, err
				}
			}
		}
	}
	var tok *checkpoint.Tokenizer
	if cursor < len(a.Records) {
		if a.Records[cursor].DType != 4 {
			return nil, nil, fmt.Errorf("needle: unexpected extra tensor")
		}
		var err error
		tok, err = checkpoint.ParseTokenizer(a.Records[cursor].Raw)
		if err != nil {
			return nil, nil, err
		}
		pad, _, _, _ := tok.SpecialIDs()
		c.PadID = pad
		if tok.VocabSize() > c.VocabSize {
			return nil, nil, fmt.Errorf("needle: tokenizer larger than model vocabulary")
		}
		cursor++
	}
	if cursor != len(a.Records) {
		return nil, nil, fmt.Errorf("needle: unexpected trailing archive records")
	}
	c.ArchiveDecoded = true
	c.ArchiveKVWindow = int(h[3])
	if slices.Equal(perms[0], numpyPermutation(n, 11, false)) && slices.Equal(perms[1], numpyPermutation(n, 13, false)) {
	} else if slices.Equal(perms[0], numpyPermutation(n, 11, true)) && slices.Equal(perms[1], numpyPermutation(n, 13, true)) {
		c.LadderWidths = []int{d}
	} else {
		return nil, nil, fmt.Errorf("needle: unknown archive Hadamard permutation")
	}
	cp.Config, _ = json.Marshal(c)
	m, err := New(cp)
	if err != nil {
		return nil, nil, err
	}
	m.p1, m.p2 = perms[0], perms[1]
	m.deployed = true
	m.archiveWindow = int(h[3])
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		if _, ok := m.tensors[string(kind)+"_head/probes"]; ok {
			if _, _, _, err = m.headGeometry(kind); err != nil {
				return nil, nil, err
			}
		}
	}
	return m, tok, nil
}
