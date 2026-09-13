package omnivoice

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
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
