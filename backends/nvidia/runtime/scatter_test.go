package nvidia

import "testing"

func TestScatterWeightedRows(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	dstBuf, err := Malloc(3 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer dstBuf.Free()
	srcBuf, err := Malloc(2 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer srcBuf.Free()
	posBuf, err := MallocBytes(2 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer posBuf.Free()
	if err := dstBuf.Upload(make([]float32, 12)); err != nil {
		t.Fatal(err)
	}
	if err := srcBuf.Upload([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatal(err)
	}
	if err := posBuf.UploadUint32([]uint32{2, 0}); err != nil {
		t.Fatal(err)
	}
	weightsBuf, err := Malloc(2)
	if err != nil {
		t.Fatal(err)
	}
	defer weightsBuf.Free()
	if err := weightsBuf.Upload([]float32{0.5, 0.5}); err != nil {
		t.Fatal(err)
	}
	if err := ScatterWeightedRowsBatch(dstBuf, srcBuf, posBuf, weightsBuf, 2, 4); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 12)
	if err := dstBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{2.5, 3, 3.5, 4, 0, 0, 0, 0, 0.5, 1, 1.5, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
}
