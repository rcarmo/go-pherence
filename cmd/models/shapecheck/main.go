package main

import (
	"fmt"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"io"
	"math"
	"os"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "shapecheck:", err)
		os.Exit(1)
	}
}
func run(args []string, out io.Writer) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: shapecheck MODEL.gguf")
	}
	g, err := gguf.Open(args[0])
	if err != nil {
		return err
	}
	defer g.Close()
	names := []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
		"output_norm.weight", "token_embd.weight",
	}
	for _, n := range names {
		t, ok := g.TensorByName(n)
		if !ok {
			fmt.Fprintf(out, "%-35s NOT FOUND\n", n)
			continue
		}
		f32, err := g.DequantF32(t)
		if err != nil {
			return fmt.Errorf("%s: %w", n, err)
		}
		if len(f32) == 0 {
			return fmt.Errorf("empty tensor %s", n)
		}
		var ss float64
		for _, v := range f32 {
			ss += float64(v) * float64(v)
		}
		rms := math.Sqrt(ss / float64(len(f32)))
		fmt.Fprintf(out, "%-35s shape=%v qt=%d n=%d rms=%.4f first=%v\n", n, t.Shape, t.QType, len(f32), rms, f32[:min(4, len(f32))])
	}
	return nil
}
