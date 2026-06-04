package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	g, _ := gguf.Open(os.Args[1])
	tokID := 12522

	t, _ := g.TensorByName("token_embd.weight")
	fmt.Fprintf(os.Stderr, "token_embd shape: %v qtype: %d\n", t.Shape, t.QType)

	f32, _ := g.DequantF32(t)
	nEmbd := int(t.Shape[0])
	nVocab := int(t.Shape[1])
	fmt.Printf("nEmbd=%d nVocab=%d total=%d\n", nEmbd, nVocab, len(f32))
	fmt.Printf("embed[%d][0:8]:", tokID)
	for k := 0; k < 8; k++ {
		fmt.Printf(" %.5f", f32[tokID*nEmbd+k])
	}
	fmt.Println()

	// Print raw Q6_K block for this token
	raw, _ := g.Raw(t)
	blockIdx := (tokID * nEmbd) / 256
	dBits := binary.LittleEndian.Uint16(raw[blockIdx*210+208:])
	fmt.Printf("Q6K block %d: d_f16=0x%04x sc[0:4]:", blockIdx, dBits)
	for i := 0; i < 4; i++ {
		fmt.Printf(" 0x%02x", raw[blockIdx*210+192+i])
	}
	fmt.Println()
}
