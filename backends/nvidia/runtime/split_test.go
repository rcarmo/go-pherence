package nvidia

import "testing"

func TestSplitGateUpBuffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	batch, inter := 2, 3
	src := []float32{1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60}
	srcBuf, err := Malloc(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer srcBuf.Free()
	gateBuf, err := Malloc(batch * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer gateBuf.Free()
	upBuf, err := Malloc(batch * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer upBuf.Free()
	if err := srcBuf.Upload(src); err != nil {
		t.Fatal(err)
	}
	if err := SplitGateUpBuffer(srcBuf, gateBuf, upBuf, batch, inter); err != nil {
		t.Fatal(err)
	}
	gate := make([]float32, batch*inter)
	up := make([]float32, batch*inter)
	if err := gateBuf.Download(gate); err != nil {
		t.Fatal(err)
	}
	if err := upBuf.Download(up); err != nil {
		t.Fatal(err)
	}
	wantG := []float32{1, 2, 3, 4, 5, 6}
	wantU := []float32{10, 20, 30, 40, 50, 60}
	for i := range wantG {
		if gate[i] != wantG[i] || up[i] != wantU[i] {
			t.Fatalf("i=%d gate=%v up=%v", i, gate, up)
		}
	}
}
