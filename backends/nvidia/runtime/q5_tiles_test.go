package nvidia

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestQ5PackedMMQ64TilesMatchBatch4(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	if fnQ5PackedMMQ64J8 == 0 || fnQ5PackedMMQ64J16 == 0 || fnQ5PackedMMQ64J24 == 0 || fnQ5Staged24 == 0 || fnQ5Staged32 == 0 {
		t.Fatal("Q5_K MMQ64 tile functions were not loaded")
	}
	for _, tc := range []struct {
		name          string
		outDim, batch int
	}{
		{name: "j8_batch_tail", outDim: 1024, batch: 7},
		{name: "j16_batch_tail", outDim: 2304, batch: 15},
		{name: "j24_batch_tail", outDim: 2304, batch: 23},
		{name: "packed_j16_tail", outDim: 2304, batch: 129},
		{name: "staged_narrow_double_tail", outDim: 79, batch: 131},
		{name: "staged_wide_double_tail", outDim: 2305, batch: 25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testQ5PackedMMQ64Case(t, tc.outDim, tc.batch)
		})
	}
}

func testQ5PackedMMQ64Case(t *testing.T, outDim, batch int) {
	t.Helper()
	testQ5PackedWidth(t, 256, outDim, batch)
}

func TestQ5StagedChunkTails(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, inDim := range []int{512, 768, 3840} {
		for _, batch := range []int{25, 131} {
			t.Run(fmt.Sprintf("k%d_b%d", inDim, batch), func(t *testing.T) { testQ5PackedWidth(t, inDim, 2305, batch) })
		}
	}
}

func testQ5PackedWidth(t *testing.T, inDim, outDim, batch int) {
	t.Helper()
	raw := make([]byte, outDim*(inDim/256)*176)
	for r := 0; r < len(raw)/176; r++ {
		block := raw[r*176 : (r+1)*176]
		binary.LittleEndian.PutUint16(block[0:2], half.F32ToF16(.02))
		binary.LittleEndian.PutUint16(block[2:4], half.F32ToF16(.003))
		for i := 0; i < 12; i++ {
			block[4+i] = byte(1 + (i+r)%12)
		}
		for i := 0; i < 32; i++ {
			block[16+i] = byte((i*5 + r*7 + 1) & 0xff)
		}
		for i := 0; i < 128; i++ {
			block[48+i] = byte((i*7 + r*11) & 0xff)
		}
	}
	m, err := UploadQ5KMatrixRows(raw, inDim, outDim)
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
	old8, old16, old24 := fnQ5PackedMMQ64J8, fnQ5PackedMMQ64J16, fnQ5PackedMMQ64J24
	oldStaged24, oldStaged32 := fnQ5Staged24, fnQ5Staged32
	defer func() {
		fnQ5Staged24, fnQ5Staged32 = oldStaged24, oldStaged32
		fnQ5PackedMMQ64J8, fnQ5PackedMMQ64J16, fnQ5PackedMMQ64J24 = old8, old16, old24
	}()
	fnQ5Staged24, fnQ5Staged32 = 0, 0
	fnQ5PackedMMQ64J8, fnQ5PackedMMQ64J16, fnQ5PackedMMQ64J24 = 0, 0, 0
	if err := GemvQ5KBatchToBuffer(wantBuf, x, batch, m); err != nil {
		t.Fatal(err)
	}
	fnQ5PackedMMQ64J8, fnQ5PackedMMQ64J16, fnQ5PackedMMQ64J24 = old8, old16, old24
	fnQ5Staged24, fnQ5Staged32 = oldStaged24, oldStaged32
	if err := GemvQ5KBatchToBuffer(gotBuf, x, batch, m); err != nil {
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
		if delta := math.Abs(float64(got[i] - want[i])); delta > 2e-4 {
			t.Fatal(fmt.Errorf("index=%d got=%g want=%g diff=%g", i, got[i], want[i], delta))
		}
	}
}
