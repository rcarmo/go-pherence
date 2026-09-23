package nvidia

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestQ6PackedMMQTilesMatchMMQ8(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	if fnQ6PackedMMQ64 == 0 || fnQ6PackedMMQ64J12 == 0 || fnQ6PackedMMQ64J16 == 0 || fnQ6Staged24 == 0 {
		t.Fatal("Q6_K MMQ64 tile functions were not loaded")
	}
	for _, tc := range []struct {
		name                 string
		inDim, outDim, batch int
	}{
		{name: "narrow_j8", inDim: 256, outDim: 256, batch: 24},
		{name: "wide_j12_tail", inDim: 8192, outDim: 64, batch: 23},
		{name: "wide_j16_tail", inDim: 8192, outDim: 64, batch: 15},
		{name: "packed_narrow_j12_tail", inDim: 256, outDim: 128, batch: 127},
		{name: "packed_wide_j16_tail", inDim: 8192, outDim: 64, batch: 257},
		{name: "staged_narrow_row_tail", inDim: 3840, outDim: 79, batch: 131},
		{name: "staged_wide_double_tail", inDim: 8192, outDim: 79, batch: 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testQ6PackedMMQCase(t, tc.inDim, tc.outDim, tc.batch)
		})
	}
}

func testQ6PackedMMQCase(t *testing.T, inDim, outDim, batch int) {
	t.Helper()
	blocks := inDim / 256
	raw := make([]byte, outDim*blocks*210)
	for r := 0; r < outDim; r++ {
		for b := 0; b < blocks; b++ {
			block := raw[(r*blocks+b)*210 : (r*blocks+b+1)*210]
			for i := 0; i < 128; i++ {
				block[i] = byte((i*7 + r*11 + b*13) & 0xff)
			}
			for i := 128; i < 192; i++ {
				block[i] = byte((i*5 + r*3 + b*17 + 1) & 0xff)
			}
			for i := 192; i < 208; i++ {
				block[i] = byte(int8((i+r+b)%31 - 15))
			}
			binary.LittleEndian.PutUint16(block[208:210], half.F32ToF16(.02))
		}
	}
	m, err := UploadQ6KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x, err := Malloc(batch * inDim)
	if err != nil {
		t.Fatal(err)
	}
	wantBuf, err := Malloc(batch * outDim)
	if err != nil {
		t.Fatal(err)
	}
	gotBuf, err := Malloc(batch * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	defer wantBuf.Free()
	defer gotBuf.Free()
	host := make([]float32, batch*inDim)
	for i := range host {
		host[i] = float32((i%31)-15) / 17
	}
	if err := x.Upload(host); err != nil {
		t.Fatal(err)
	}
	old64, old12, old16, oldStaged := fnQ6PackedMMQ64, fnQ6PackedMMQ64J12, fnQ6PackedMMQ64J16, fnQ6Staged24
	defer func() {
		fnQ6PackedMMQ64, fnQ6PackedMMQ64J12, fnQ6PackedMMQ64J16, fnQ6Staged24 = old64, old12, old16, oldStaged
	}()
	fnQ6PackedMMQ64, fnQ6PackedMMQ64J12, fnQ6PackedMMQ64J16, fnQ6Staged24 = 0, 0, 0, 0
	if err := GemvQ6KBatchToBuffer(wantBuf, x, batch, m); err != nil {
		t.Fatal(err)
	}
	fnQ6PackedMMQ64, fnQ6PackedMMQ64J12, fnQ6PackedMMQ64J16, fnQ6Staged24 = old64, old12, old16, oldStaged
	if err := GemvQ6KBatchToBuffer(gotBuf, x, batch, m); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	want, got := make([]float32, batch*outDim), make([]float32, batch*outDim)
	if err := wantBuf.Download(want); err != nil {
		t.Fatal(err)
	}
	if err := gotBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		limit := 2e-4 * math.Max(1, math.Abs(float64(want[i])))
		if delta > limit {
			t.Fatal(fmt.Errorf("index=%d got=%g want=%g diff=%g limit=%g", i, got[i], want[i], delta, limit))
		}
	}
}
