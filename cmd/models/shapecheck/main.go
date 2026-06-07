package main

import (
	"fmt"
	"math"
	"os"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	g, _ := gguf.Open(os.Args[1])
	names := []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
		"output_norm.weight", "token_embd.weight",
	}
	for _, n := range names {
		t, ok := g.TensorByName(n)
		if !ok { fmt.Printf("%-35s NOT FOUND\n", n); continue }
		f32, _ := g.DequantF32(t)
		var ss float64
		for _, v := range f32 { ss += float64(v)*float64(v) }
		rms := math.Sqrt(ss / float64(len(f32)))
		fmt.Printf("%-35s shape=%v qt=%d n=%d rms=%.4f first4=[%.4f %.4f %.4f %.4f]\n",
			n, t.Shape, t.QType, len(f32), rms, f32[0], f32[1], f32[2], f32[3])
	}
}
