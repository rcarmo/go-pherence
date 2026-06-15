package nvidia

import (
	"math"
	"testing"
)

func TestExpertMetaReduce(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	off, err := MallocBytes(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer off.Free()
	w, err := Malloc(5)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Free()
	counts, err := MallocBytes(3 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer counts.Free()
	sums, err := Malloc(3)
	if err != nil {
		t.Fatal(err)
	}
	defer sums.Free()
	if err := off.UploadUint32([]uint32{0, 2, 2, 5}); err != nil {
		t.Fatal(err)
	}
	if err := w.Upload([]float32{0.5, 0.25, 9, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := ExpertMetaReduce(off, w, counts, sums, 3); err != nil {
		t.Fatal(err)
	}
	gotC := make([]byte, 12)
	if err := counts.DownloadBytes(gotC); err != nil {
		t.Fatal(err)
	}
	gotS := make([]float32, 3)
	if err := sums.Download(gotS); err != nil {
		t.Fatal(err)
	}
	wantC := []byte{2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0}
	for i := range wantC {
		if gotC[i] != wantC[i] {
			t.Fatalf("counts bytes=%v", gotC)
		}
	}
	wantS := []float32{0.75, 0, 12}
	for i := range wantS {
		if math.Abs(float64(gotS[i]-wantS[i])) > 1e-6 {
			t.Fatalf("sums=%v want=%v", gotS, wantS)
		}
	}
}
