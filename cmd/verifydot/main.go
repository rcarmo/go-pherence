package main
import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"github.com/rcarmo/go-pherence/loader/gguf"
)
func f16(h uint16) float32 {
	if h == 0 { return 0 }
	sign := uint32(h>>15)&1; exp := uint32(h>>10)&0x1f; mant := uint32(h)&0x3ff
	if exp == 0 { for mant&0x400==0 { mant<<=1; exp-- }; exp++; mant&=0x3ff }
	exp += 127-15
	bits := (sign<<31)|(exp<<23)|(mant<<13)
	return math.Float32frombits(bits)
}
func main() {
	g, _ := gguf.Open(os.Args[1])
	tokT, _ := g.TensorByName("token_embd.weight")
	raw, _ := g.Raw(tokT)
	fmt.Printf("Raw bytes for token_embd: %d bytes\n", len(raw))
	
	// First Q2_K block (84 bytes, 256 elements of row 0)
	blk := raw[0:84]
	scales := blk[0:16]
	qs := blk[16:80]
	d := f16(binary.LittleEndian.Uint16(blk[80:82]))
	dmin := f16(binary.LittleEndian.Uint16(blk[82:84]))
	
	fmt.Printf("Block 0: d=%.8f dmin=%.8f\n", d, dmin)
	fmt.Printf("  scales[0:4]: %d %d %d %d\n", scales[0], scales[1], scales[2], scales[3])
	fmt.Printf("  qs[0:4]: %02x %02x %02x %02x\n", qs[0], qs[1], qs[2], qs[3])
	
	// Manual dequant of first 8 elements
	sc0 := scales[0]
	dl := d * float32(sc0&0x0F)
	ml := dmin * float32(sc0>>4)
	fmt.Printf("  sc0=0x%02x dl=%.8f ml=%.8f\n", sc0, dl, ml)
	for i := 0; i < 8; i++ {
		q := qs[i/4]
		shift := uint(2 * (i % 4))
		val := (q >> shift) & 3
		result := dl*float32(val) - ml
		fmt.Printf("  elem[%d]: q_byte=0x%02x shift=%d val=%d → %.8f\n", i, q, shift, val, result)
	}
	
	// Compare with DequantF32 output
	tokF32, _ := g.DequantF32(tokT)
	fmt.Printf("\n  DequantF32[0:8]:")
	for i := 0; i < 8; i++ { fmt.Printf(" %.8f", tokF32[i]) }
	fmt.Println()
}
