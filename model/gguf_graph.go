package model

import gograph "github.com/rcarmo/go-pherence/runtime/graph"

// BuildDecodeGraph builds the default planned graph for one Tiny/LLaMA decode step.
//
// This graph is a backend-neutral contract: execution can be lowered to the
// current Go/RVV executor, GGML, ORT, Vulkan, or libllama-style schedulers.
func (m *GGUFLlama) BuildDecodeGraph() (*gograph.Graph, *gograph.Plan, error) {
	cfg := m.Config
	h := cfg.HiddenSize
	nH := cfg.NumHeads
	nKV := cfg.NumKVHeads
	hd := cfg.HeadDim
	kvDim := nKV * hd
	ffn := cfg.FFNHiddenSize

	g := gograph.New("gguf_llama_decode")
	tok := g.AddValue("token_id", gograph.Shape{1}, gograph.I32, true)
	pos := g.AddValue("position", gograph.Shape{1}, gograph.I32, true)
	_ = pos
	hidden := g.AddValue("hidden", gograph.Shape{h}, gograph.F32, false)
	g.AddNode("token_embedding", gograph.OpEmbedding, []gograph.ValueID{tok}, []gograph.ValueID{hidden}, map[string]any{"table": "token_embd.weight"})

	for il := 0; il < cfg.NumLayers; il++ {
		prefix := "blk."
		attnIn := g.AddValue(prefix+itoa(il)+".attn_normed", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".attn_norm", gograph.OpRMSNorm, []gograph.ValueID{hidden}, []gograph.ValueID{attnIn}, map[string]any{"weight": prefix + itoa(il) + ".attn_norm.weight", "eps": cfg.RMSNormEps})

		q := g.AddValue(prefix+itoa(il)+".q", gograph.Shape{nH * hd}, gograph.F32, false)
		k := g.AddValue(prefix+itoa(il)+".k", gograph.Shape{kvDim}, gograph.F32, false)
		v := g.AddValue(prefix+itoa(il)+".v", gograph.Shape{kvDim}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".q_proj", gograph.OpMatMul, []gograph.ValueID{attnIn}, []gograph.ValueID{q}, map[string]any{"weight": prefix + itoa(il) + ".attn_q.weight", "in": h, "out": nH * hd})
		g.AddNode(prefix+itoa(il)+".k_proj", gograph.OpMatMul, []gograph.ValueID{attnIn}, []gograph.ValueID{k}, map[string]any{"weight": prefix + itoa(il) + ".attn_k.weight", "in": h, "out": kvDim})
		g.AddNode(prefix+itoa(il)+".v_proj", gograph.OpMatMul, []gograph.ValueID{attnIn}, []gograph.ValueID{v}, map[string]any{"weight": prefix + itoa(il) + ".attn_v.weight", "in": h, "out": kvDim})

		qr := g.AddValue(prefix+itoa(il)+".q_rope", gograph.Shape{nH * hd}, gograph.F32, false)
		kr := g.AddValue(prefix+itoa(il)+".k_rope", gograph.Shape{kvDim}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".q_rope", gograph.OpRoPE, []gograph.ValueID{q}, []gograph.ValueID{qr}, map[string]any{"heads": nH, "head_dim": hd, "rot_half": m.rotHalf})
		g.AddNode(prefix+itoa(il)+".k_rope", gograph.OpRoPE, []gograph.ValueID{k}, []gograph.ValueID{kr}, map[string]any{"heads": nKV, "head_dim": hd, "rot_half": m.rotHalf})
		g.AddNode(prefix+itoa(il)+".kv_write", gograph.OpKVWrite, []gograph.ValueID{kr, v}, nil, map[string]any{"layer": il})

		attnOut := g.AddValue(prefix+itoa(il)+".attn_out", gograph.Shape{nH * hd}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".attention", gograph.OpAttention, []gograph.ValueID{qr}, []gograph.ValueID{attnOut}, map[string]any{"layer": il, "heads": nH, "kv_heads": nKV, "head_dim": hd})
		o := g.AddValue(prefix+itoa(il)+".o", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".o_proj", gograph.OpMatMul, []gograph.ValueID{attnOut}, []gograph.ValueID{o}, map[string]any{"weight": prefix + itoa(il) + ".attn_output.weight", "in": nH * hd, "out": h})
		h2 := g.AddValue(prefix+itoa(il)+".attn_resid", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".attn_residual", gograph.OpAdd, []gograph.ValueID{hidden, o}, []gograph.ValueID{h2}, nil)

		ffnIn := g.AddValue(prefix+itoa(il)+".ffn_normed", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".ffn_norm", gograph.OpRMSNorm, []gograph.ValueID{h2}, []gograph.ValueID{ffnIn}, map[string]any{"weight": prefix + itoa(il) + ".ffn_norm.weight", "eps": cfg.RMSNormEps})
		gate := g.AddValue(prefix+itoa(il)+".gate", gograph.Shape{ffn}, gograph.F32, false)
		up := g.AddValue(prefix+itoa(il)+".up", gograph.Shape{ffn}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".gate_proj", gograph.OpMatMul, []gograph.ValueID{ffnIn}, []gograph.ValueID{gate}, map[string]any{"weight": prefix + itoa(il) + ".ffn_gate.weight", "in": h, "out": ffn})
		g.AddNode(prefix+itoa(il)+".up_proj", gograph.OpMatMul, []gograph.ValueID{ffnIn}, []gograph.ValueID{up}, map[string]any{"weight": prefix + itoa(il) + ".ffn_up.weight", "in": h, "out": ffn})
		silu := g.AddValue(prefix+itoa(il)+".silu", gograph.Shape{ffn}, gograph.F32, false)
		mid := g.AddValue(prefix+itoa(il)+".ffn_mid", gograph.Shape{ffn}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".silu", gograph.OpSiLU, []gograph.ValueID{gate}, []gograph.ValueID{silu}, nil)
		g.AddNode(prefix+itoa(il)+".silu_mul", gograph.OpMul, []gograph.ValueID{silu, up}, []gograph.ValueID{mid}, nil)
		down := g.AddValue(prefix+itoa(il)+".down", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".down_proj", gograph.OpMatMul, []gograph.ValueID{mid}, []gograph.ValueID{down}, map[string]any{"weight": prefix + itoa(il) + ".ffn_down.weight", "in": ffn, "out": h})
		nextHidden := g.AddValue(prefix+itoa(il)+".ffn_resid", gograph.Shape{h}, gograph.F32, false)
		g.AddNode(prefix+itoa(il)+".ffn_residual", gograph.OpAdd, []gograph.ValueID{h2, down}, []gograph.ValueID{nextHidden}, nil)
		hidden = nextHidden
	}

	final := g.AddValue("final_normed", gograph.Shape{h}, gograph.F32, false)
	g.AddNode("final_norm", gograph.OpRMSNorm, []gograph.ValueID{hidden}, []gograph.ValueID{final}, map[string]any{"weight": "output_norm.weight", "eps": cfg.RMSNormEps})
	logits := g.AddValue("logits", gograph.Shape{cfg.VocabSize}, gograph.F32, false)
	g.AddNode("lm_head", gograph.OpMatMul, []gograph.ValueID{final}, []gograph.ValueID{logits}, map[string]any{"weight": "output.weight", "in": h, "out": cfg.VocabSize})

	p, err := gograph.BuildPlan(g)
	return g, p, err
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	p := len(buf)
	for i > 0 {
		p--
		buf[p] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[p:])
}
