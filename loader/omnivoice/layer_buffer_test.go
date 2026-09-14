package omnivoice

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestLayerBufferReuse(t *testing.T) {
	cfg := sampleConfig(t)
	dir := t.TempDir()
	raw, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	specs, _ := expectedTensorSpecs(cfg)
	writeSyntheticSafetensors(t, filepath.Join(dir, checkpointFileName), specs)
	w, err := OpenWeights(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	buf, err := w.NewLayerBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if err = buf.Load(w, 0); err != nil {
		t.Fatal(err)
	}
	n := testing.AllocsPerRun(100, func() { err = buf.Load(w, 0) })
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("allocs=%g", n)
	}
	if err = buf.Load(w, -1); err == nil {
		t.Fatal("accepted negative layer")
	}
	if len(buf.Tensors) != 11 || buf.Bytes() == 0 {
		t.Fatal("missing tensors")
	}
}
func TestConvertInto(t *testing.T) {
	for _, tc := range []struct {
		dtype string
		bits  uint32
		size  int
	}{{"F16", 0x3e00, 2}, {"BF16", 0x3fc0, 2}, {"F32", math.Float32bits(1.5), 4}} {
		raw := make([]byte, tc.size)
		if tc.size == 2 {
			binary.LittleEndian.PutUint16(raw, uint16(tc.bits))
		} else {
			binary.LittleEndian.PutUint32(raw, tc.bits)
		}
		dst := make([]float32, 1)
		if err := convertInto(dst, raw, tc.dtype); err != nil {
			t.Fatal(err)
		}
		if dst[0] != 1.5 {
			t.Fatalf("%s = %g", tc.dtype, dst[0])
		}
	}
	if err := convertInto(make([]float32, 1), []byte{0}, "F16"); err == nil {
		t.Fatal("bad size accepted")
	}
}

func TestConvertIntoF16AllPatterns(t *testing.T) {
	raw := make([]byte, 0x10000*2)
	dst := make([]float32, 0x10000)
	for i := range dst {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(i))
	}
	if err := convertInto(dst, raw, "F16"); err != nil {
		t.Fatal(err)
	}
	for i := range dst {
		want := half.F16ToF32(uint16(i))
		if math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("pattern 0x%04x got=%08x want=%08x", i, math.Float32bits(dst[i]), math.Float32bits(want))
		}
	}
}

func BenchmarkConvertIntoF16(b *testing.B) {
	const n = 3584
	raw := make([]byte, n*2)
	dst := make([]float32, n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(raw[i*2:], half.F32ToF16(float32(i%257)*0.03125-4))
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := convertInto(dst, raw, "F16"); err != nil {
			b.Fatal(err)
		}
	}
}
