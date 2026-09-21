package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "embcheck:", err)
		os.Exit(1)
	}
}
func run(args []string, out, errOut io.Writer) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: embcheck MODEL.gguf")
	}
	g, err := gguf.Open(args[0])
	if err != nil {
		return err
	}
	defer g.Close()
	tokID := 12522

	t, ok := g.TensorByName("token_embd.weight")
	if !ok {
		return fmt.Errorf("missing token_embd.weight")
	}
	if len(t.Shape) != 2 || t.Shape[0] < 8 || uint64(tokID) >= t.Shape[1] {
		return fmt.Errorf("embedding shape %v cannot supply token %d width8", t.Shape, tokID)
	}
	fmt.Fprintf(errOut, "token_embd shape: %v qtype: %d\n", t.Shape, t.QType)

	f32, err := g.DequantF32(t)
	if err != nil {
		return err
	}
	nEmbd := int(t.Shape[0])
	nVocab := int(t.Shape[1])
	fmt.Fprintf(out, "nEmbd=%d nVocab=%d total=%d\n", nEmbd, nVocab, len(f32))
	fmt.Fprintf(out, "embed[%d][0:8]:", tokID)
	for k := 0; k < 8; k++ {
		fmt.Fprintf(out, " %.5f", f32[tokID*nEmbd+k])
	}
	fmt.Fprintln(out)

	// Print raw Q6_K block for this token
	if t.QType != gguf.QuantQ6_K {
		return nil
	}
	raw, err := g.Raw(t)
	if err != nil {
		return err
	}
	blockIdx := (tokID * nEmbd) / 256
	if blockIdx >= len(raw)/210 {
		return fmt.Errorf("short Q6_K embedding block")
	}
	dBits := binary.LittleEndian.Uint16(raw[blockIdx*210+208:])
	fmt.Fprintf(out, "Q6K block %d: d_f16=0x%04x sc[0:4]:", blockIdx, dBits)
	for i := 0; i < 4; i++ {
		fmt.Fprintf(out, " 0x%02x", raw[blockIdx*210+192+i])
	}
	fmt.Fprintln(out)
	return nil
}
