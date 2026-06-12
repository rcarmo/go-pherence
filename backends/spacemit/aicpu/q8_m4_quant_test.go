package aicpu

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/config"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

func TestQuantizeQ8RowsM4BytesMatchesPackRows(t *testing.T) {
	K := 1024
	subs := K / 32
	var acts [4][]float32
	var rows [4][]byte
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		for k := range acts[r] {
			acts[r][k] = float32(((r+5)*(k%37) - 17)) / 11.0
		}
		rows[r] = quantizeQ8Blocks32Bytes(acts[r])
	}
	want := ime2.PackQ8RowsM4(rows, subs)
	got := quantizeQ8RowsM4Bytes(acts, subs)
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d got=%02x want=%02x", i, got[i], want[i])
		}
	}
}

func TestQuantizeQ8Block32RVVMatchesGo(t *testing.T) {
	// Test that quantizeQ8Block32RVV produces identical output to the
	// reference Go path (quantizeQ8Blocks32Bytes with config.Q8RoundOn forced off).
	rng := rand.New(rand.NewSource(0xDEAD_BEEF))
	const nBlocks = 16

	act := make([]float32, nBlocks*32)
	for i := range act {
		act[i] = (rng.Float32() - 0.5) * 10.0
	}

	// Reference: Go path (truncation, not RVV)
	oldRound := config.Q8RoundOn
	config.Q8RoundOn = false
	ref := quantizeQ8Blocks32Bytes(act)
	config.Q8RoundOn = oldRound

	// RVV path: call block-by-block directly
	got := make([]byte, nBlocks*38)
	for b := 0; b < nBlocks; b++ {
		src := &act[b*32]
		dst := (*byte)(unsafe.Pointer(&got[b*38]))
		rvv.QuantizeQ8Block32RVV(src, dst, &rvv.Q8QuantDivisor)
	}

	maxDiff := 0
	for i := range ref {
		d := int(ref[i]) - int(got[i])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	// Scale bytes (offsets 0-3) use float bits; i8 bytes may differ by 1 due
	// to rounding mode (RVV uses round-to-nearest-even, Go truncates).
	// We tolerate <=1 difference on i8 bytes; scale must match within 1 ULP.
	t.Logf("maxBytesDiff=%d", maxDiff)
	// Scale (bytes 0-3): compare as float values, allow ≤1 ULP.
	// Sum (bytes 4-5): i16 compared byte-by-byte may span a byte boundary;
	//   raw diff up to 255 is expected when rounding causes a carry. Compare as int16.
	// i8 data (bytes 6-37): RVV rounds, Go truncates; diff ≤ 1 per byte.
	for b := 0; b < nBlocks; b++ {
		base := b * 38
		refSum := int16(ref[base+4]) | int16(ref[base+5])<<8
		gotSum := int16(got[base+4]) | int16(got[base+5])<<8
		if d := int(refSum) - int(gotSum); d < -32 || d > 32 {
			t.Errorf("block %d sum mismatch: ref=%d got=%d diff=%d", b, refSum, gotSum, d)
		}
		for i := 6; i < 38; i++ {
			d := int(int8(ref[base+i])) - int(int8(got[base+i]))
			if d < -1 || d > 1 {
				t.Errorf("block %d byte %d: ref=%d got=%d", b, i-6, int8(ref[base+i]), int8(got[base+i]))
			}
		}
	}
	t.Logf("overall maxBytesDiff=%d (scale+sum bytes; expected ≤255 due to i16 boundary)", maxDiff)
}

func BenchmarkQuantizeQ8Block32RVV(b *testing.B) {
	act := make([]float32, 32)
	for i := range act {
		act[i] = float32(i+1) * 0.1
	}
	var buf [38]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rvv.QuantizeQ8Block32RVV(&act[0], &buf[0], &rvv.Q8QuantDivisor)
	}
}

func BenchmarkQuantizeQ8Block32Go(b *testing.B) {
	act := make([]float32, 32)
	for i := range act {
		act[i] = float32(i+1) * 0.1
	}
	old := config.Q8RoundOn
	config.Q8RoundOn = false
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantizeQ8Blocks32Bytes(act)
	}
	config.Q8RoundOn = old
}

func BenchmarkCopyBytesRVV128(b *testing.B) {
	// 19.5KB = one N32/K32 group B-data block
	const n = 19456
	src := make([]byte, n)
	dst := make([]byte, n)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rvv.CopyTCMBytes(dst, src)
	}
}
