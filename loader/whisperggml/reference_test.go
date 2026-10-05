package whisperggml

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	gguf "github.com/rcarmo/go-pherence/loader/gguf"
	"math"
	"os"
	"testing"
)

func TestPinnedQ5IndependentDecode(t *testing.T) {
	p := os.Getenv("GO_PHERENCE_Q5_BLOCK_REFERENCE_INPUT")
	if p == "" {
		t.Skip("independent retained GGML block oracle")
	}
	in, err := os.ReadFile(p)
	if err != nil || len(in) < 22 || len(in) > 1<<20 || len(in)%22 != 0 {
		t.Fatal("input bound", err)
	}
	ref, err := os.ReadFile(os.Getenv("GO_PHERENCE_Q5_BLOCK_REFERENCE_OUTPUT"))
	if err != nil || len(ref) != len(in)/22*32*4 {
		t.Fatal("output bound", err)
	}
	for _, x := range []struct {
		bytes []byte
		pin   string
	}{{in, os.Getenv("GO_PHERENCE_Q5_BLOCK_REFERENCE_INPUT_SHA256")}, {ref, os.Getenv("GO_PHERENCE_Q5_BLOCK_REFERENCE_OUTPUT_SHA256")}} {
		sum := sha256.Sum256(x.bytes)
		if len(x.pin) != 64 || hex.EncodeToString(sum[:]) != x.pin {
			t.Fatal("oracle integrity")
		}
	}
	out, err := gguf.DequantToF32(in, gguf.QuantQ5_0, len(in)/22*32)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
		if math.Float32bits(v) != math.Float32bits(want) {
			t.Fatal("independent Q5 block bits", i, v, want)
		}
	}
	t.Logf("independent_q5_values=%d bit_exact=true", len(out))
}
