package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestQ4UpstreamMMQMatchesAcceptedQ8Projection(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const inDim, outDim, batch = 256, 256, 4
	raw := make([]byte, outDim*144)
	for r := 0; r < outDim; r++ {
		z := raw[r*144:]
		binary.LittleEndian.PutUint16(z, half.F32ToF16(.02))
		binary.LittleEndian.PutUint16(z[2:], half.F32ToF16(.003))
		for i := 0; i < 12; i++ {
			z[4+i] = byte(1 + (i+r)%12)
		}
		for i := 0; i < 128; i++ {
			z[16+i] = byte((i + r*3) & 255)
		}
	}
	accepted, err := UploadQ4KMatrixRowsCoalesced(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer accepted.Free()
	rawBuf, err := MallocBytes(len(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer rawBuf.Free()
	if err = rawBuf.UploadBytes(raw); err != nil {
		t.Fatal(err)
	}
	x, _ := Malloc(batch * inDim)
	wantBuf, _ := Malloc(batch * outDim)
	gotBuf, _ := Malloc(batch * outDim)
	defer x.Free()
	defer wantBuf.Free()
	defer gotBuf.Free()
	host := make([]float32, batch*inDim)
	for i := range host {
		host[i] = float32((i%31)-15) / 17
	}
	if err = x.Upload(host); err != nil {
		t.Fatal(err)
	}
	if err = GemmQ4CoalescedQ8ToBuffer(wantBuf, x, batch, accepted); err != nil {
		t.Fatal(err)
	}
	if err = ProbeQ4UpstreamMMQ(gotBuf, x, rawBuf, batch, inDim, outDim); err != nil {
		t.Fatal(err)
	}
	if err = SyncErr(); err != nil {
		t.Fatal(err)
	}
	want, got := make([]float32, batch*outDim), make([]float32, batch*outDim)
	wantBuf.Download(want)
	gotBuf.Download(got)
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > 0.02 {
			t.Fatalf("index=%d got=%g want=%g diff=%g", i, got[i], want[i], d)
		}
	}
}
