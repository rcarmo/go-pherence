package diffusiongemma

import "testing"

func TestDecodeFloatRowSupportsGenericF8E4M3(t *testing.T) {
	raw := []byte{0x00, 0x38, 0x40, 0xb8, 0x01}
	want := []float32{0, 1, 2, -1, 1.0 / 512.0}
	for _, dtype := range []string{"F8_E4M3", "F8_E4M3FN"} {
		got := make([]float32, len(raw))
		if err := decodeFloatRowTo(got, raw, dtype); err != nil {
			t.Fatalf("decode %s: %v", dtype, err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("decode %s byte %#02x = %g, want %g", dtype, raw[i], got[i], want[i])
			}
		}
	}
}
