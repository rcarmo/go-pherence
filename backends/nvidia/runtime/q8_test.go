package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func xBufFromFloatTest(t *testing.T, x []float32) *Buffer {
	t.Helper()
	buf, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	if err := buf.Upload(x); err != nil {
		buf.Free()
		t.Fatal(err)
	}
	t.Cleanup(buf.Free)
	return buf
}

func TestGemvQ8_0MatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim := 32, 3
	raw := make([]byte, outDim*34)
	scales := []float32{0.25, 0.5, 0.125}
	for r := 0; r < outDim; r++ {
		row := raw[r*34 : (r+1)*34]
		binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(scales[r]))
		for i := 0; i < inDim; i++ {
			row[2+i] = byte(int8((r+1)*(i%7) - 9))
		}
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i%5)-2) * 0.2
	}
	m, err := UploadQ8_0MatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	got := make([]float32, outDim)
	if err := GemvQ8_0(got, x, m); err != nil {
		t.Fatal(err)
	}
	gotBatch := make([]float32, outDim*2)
	if err := GemvQ8_0Batch(gotBatch, append(x, x...), 2, m); err != nil {
		t.Fatal(err)
	}
	dstBuf, err := Malloc(3 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dstBuf.Free()
	if err := dstBuf.Upload(make([]float32, 3*outDim)); err != nil {
		t.Fatal(err)
	}
	posBuf, err := MallocBytes(2 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer posBuf.Free()
	if err := posBuf.UploadUint32([]uint32{2, 0}); err != nil {
		t.Fatal(err)
	}
	weightBuf, err := Malloc(2)
	if err != nil {
		t.Fatal(err)
	}
	defer weightBuf.Free()
	if err := weightBuf.Upload([]float32{0.5, 0.25}); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ8_0BatchScatter(dstBuf, xBufFromFloatTest(t, append(x, x...)), posBuf, weightBuf, 2, m); err != nil {
		t.Fatal(err)
	}
	scattered := make([]float32, 3*outDim)
	if err := dstBuf.Download(scattered); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < outDim; r++ {
		var want float32
		row := raw[r*34 : (r+1)*34]
		s := half.F16ToF32(binary.LittleEndian.Uint16(row[:2]))
		for i := 0; i < inDim; i++ {
			want += s * float32(int8(row[2+i])) * x[i]
		}
		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("row=%d got=%g want=%g all=%v", r, got[r], want, got)
		}
		if math.Abs(float64(gotBatch[r]-want)) > 1e-3 || math.Abs(float64(gotBatch[outDim+r]-want)) > 1e-3 {
			t.Fatalf("batch row=%d got=%g/%g want=%g all=%v", r, gotBatch[r], gotBatch[outDim+r], want, gotBatch)
		}
		if math.Abs(float64(scattered[2*outDim+r]-want*0.5)) > 1e-3 || math.Abs(float64(scattered[r]-want*0.25)) > 1e-3 {
			t.Fatalf("scatter row=%d got dst0=%g dst2=%g want=%g", r, scattered[r], scattered[2*outDim+r], want)
		}
	}
}
