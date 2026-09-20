package nvidia

import (
	"math"
	"testing"
)

func TestWhisperKernelGeometryWithoutCUDA(t *testing.T) {
	b := &Buffer{Ptr: 1, Size: 1 << 24}
	short := &Buffer{Ptr: 1, Size: 4}
	if !validWhisperAttention(b, b, b, b, 4, 32, 2, 64, 1, "full") {
		t.Fatal("valid attention")
	}
	for _, tc := range []struct {
		sq, kv, h, d int
		name         string
	}{{4, 2049, 2, 64, "full"}, {4, 1, 2, 129, "full_online"}, {65536, 1, 1, 64, "cross"}, {4, 1, 1, 64, "unknown"}, {1, math.MaxInt32, 4, 128, "cross"}} {
		if validWhisperAttention(b, b, b, b, tc.sq, tc.kv, tc.h, tc.d, 1, tc.name) {
			t.Fatal(tc)
		}
	}
	if validWhisperAttention(b, b, short, b, 4, 32, 2, 64, 1, "full") || validWhisperAttention(b, b, b, b, 4, 32, 2, 64, float32(math.NaN()), "full") {
		t.Fatal("short/nonfinite attention")
	}
	if !validWhisperConv(b, b, b, nil, 2, 9, 3, 5, "s2") || validWhisperConv(b, b, b, nil, 2, 9, 3, 9, "s2") || validWhisperConv(b, short, b, nil, 2, 9, 3, 5, "s2") {
		t.Fatal("conv geometry")
	}
	if !validWhisperMel(b, b, b, b, 10, 400, 160, 80, 257) || validWhisperMel(b, short, b, b, 10, 400, 160, 80, 257) || validWhisperMel(b, b, b, b, 10, 513, 160, 80, 257) {
		t.Fatal("mel geometry")
	}
	if whisperF32Extent(b, int(^uint(0)>>1), 4) || whisperF32Extent(&Buffer{Size: 1024}, 1) {
		t.Fatal("bad float span")
	}
}
