package nvidia

import "testing"

func TestGatherRows(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	src := []float32{1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8}
	srcBuf, err := Malloc(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer srcBuf.Free()
	posBuf, err := MallocBytes(3 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer posBuf.Free()
	dstBuf, err := Malloc(3 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer dstBuf.Free()
	if err := srcBuf.Upload(src); err != nil {
		t.Fatal(err)
	}
	if err := posBuf.UploadUint32([]uint32{2, 0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := GatherRows(dstBuf, srcBuf, posBuf, 3, 4); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 12)
	if err := dstBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{5, 6, 7, 8, 1, 2, 3, 4, 10, 20, 30, 40}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
}
