package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGemvQ8_0ScatterByWork(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, active := 32, 3, 2
	raw := make([]byte, active*outDim*34)
	for r := 0; r < active*outDim; r++ {
		row := raw[r*34 : (r+1)*34]
		binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(0.1+float32(r)*0.02))
		for i := 0; i < inDim; i++ {
			row[2+i] = byte(int8((r+i)%13 - 6))
		}
	}
	m, err := UploadQ8_0MatrixRows(raw, inDim, outDim*active)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	workLen := 3
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.1
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	wa, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wa.Free()
	if err := wa.UploadUint32([]uint32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	wp, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wp.Free()
	if err := wp.UploadUint32([]uint32{2, 0, 1}); err != nil {
		t.Fatal(err)
	}
	ww, err := Malloc(workLen)
	if err != nil {
		t.Fatal(err)
	}
	defer ww.Free()
	if err := ww.Upload([]float32{0.5, 0.25, 1}); err != nil {
		t.Fatal(err)
	}
	dst, err := Malloc(3 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()
	if err := dst.Upload(make([]float32, 3*outDim)); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ8_0ScatterByWork(dst, xb, wa, wp, ww, workLen, active, m); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 3*outDim)
	if err := dst.Download(got); err != nil {
		t.Fatal(err)
	}
	for w := 0; w < workLen; w++ {
		activeIdx := []int{0, 1, 0}[w]
		pos := []int{2, 0, 1}[w]
		weight := []float32{0.5, 0.25, 1}[w]
		tmp := make([]float32, outDim)
		sub := &GPUQ8_0Matrix{Q: m.Q, Scales: m.Scales, InDim: inDim, OutDim: outDim * active}
		_ = sub
		for r := 0; r < outDim; r++ {
			row := raw[(activeIdx*outDim+r)*34 : (activeIdx*outDim+r+1)*34]
			s := half.F16ToF32(binary.LittleEndian.Uint16(row[:2]))
			var sum float32
			for i := 0; i < inDim; i++ {
				sum += s * float32(int8(row[2+i])) * x[w*inDim+i]
			}
			tmp[r] = sum * weight
		}
		for r := 0; r < outDim; r++ {
			if math.Abs(float64(got[pos*outDim+r]-tmp[r])) > 1e-3 {
				t.Fatalf("got=%v expected pos=%d row=%d %g", got, pos, r, tmp[r])
			}
		}
	}
}

func TestGemvQ8_0ScatterByWorkPtrs(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, active := 32, 3, 2
	raw := make([]byte, active*outDim*34)
	for r := 0; r < active*outDim; r++ {
		row := raw[r*34 : (r+1)*34]
		binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(0.1+float32(r)*0.02))
		for i := 0; i < inDim; i++ {
			row[2+i] = byte(int8((r+i)%13 - 6))
		}
	}
	mats := make([]*GPUQ8_0Matrix, active)
	for a := 0; a < active; a++ {
		m, err := UploadQ8_0MatrixRows(raw[a*outDim*34:(a+1)*outDim*34], inDim, outDim)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Free()
		mats[a] = m
	}
	table, err := UploadQ8_0PointerTable(mats)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Free()
	workLen := 3
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.1
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	wa, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wa.Free()
	if err := wa.UploadUint32([]uint32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	wp, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wp.Free()
	if err := wp.UploadUint32([]uint32{2, 0, 1}); err != nil {
		t.Fatal(err)
	}
	ww, err := Malloc(workLen)
	if err != nil {
		t.Fatal(err)
	}
	defer ww.Free()
	if err := ww.Upload([]float32{0.5, 0.25, 1}); err != nil {
		t.Fatal(err)
	}
	dst, err := Malloc(3 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()
	if err := dst.Upload(make([]float32, 3*outDim)); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ8_0ScatterByWorkPtrs(dst, xb, wa, wp, ww, workLen, table); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 3*outDim)
	if err := dst.Download(got); err != nil {
		t.Fatal(err)
	}
	for w := 0; w < workLen; w++ {
		activeIdx := []int{0, 1, 0}[w]
		pos := []int{2, 0, 1}[w]
		weight := []float32{0.5, 0.25, 1}[w]
		for r := 0; r < outDim; r++ {
			row := raw[(activeIdx*outDim+r)*34 : (activeIdx*outDim+r+1)*34]
			s := half.F16ToF32(binary.LittleEndian.Uint16(row[:2]))
			var want float32
			for i := 0; i < inDim; i++ {
				want += s * float32(int8(row[2+i])) * x[w*inDim+i]
			}
			want *= weight
			if math.Abs(float64(got[pos*outDim+r]-want)) > 1e-3 {
				t.Fatalf("got=%v expected pos=%d row=%d %g", got, pos, r, want)
			}
		}
	}
}

func TestUploadQ8_0MatrixRowsRejectsInvalidInput(t *testing.T) {
	if _, err := UploadQ8_0MatrixRows(nil, 31, 1); err == nil {
		t.Fatal("accepted non-32-aligned Q8 input dimension")
	}
	if _, err := UploadQ8_0MatrixRows(make([]byte, 33), 32, 1); err == nil {
		t.Fatal("accepted short Q8 raw row")
	}
	if _, err := UploadQ8_0MatrixRows(make([]byte, 34), 32, 0); err == nil {
		t.Fatal("accepted zero Q8 output dimension")
	}
}

func TestUploadQ8_0PointerTableRejectsInvalidMatrices(t *testing.T) {
	validBuf := &Buffer{Ptr: 1, Size: 64}
	valid := &GPUQ8_0Matrix{Q: validBuf, Scales: validBuf, InDim: 32, OutDim: 4}
	if _, err := UploadQ8_0PointerTable(nil); err == nil {
		t.Fatal("accepted empty Q8_0 pointer table")
	}
	if _, err := UploadQ8_0PointerTable([]*GPUQ8_0Matrix{valid, nil}); err == nil {
		t.Fatal("accepted nil Q8_0 matrix")
	}
	badDims := *valid
	badDims.InDim = 64
	if _, err := UploadQ8_0PointerTable([]*GPUQ8_0Matrix{valid, &badDims}); err == nil {
		t.Fatal("accepted mismatched Q8_0 matrix dimensions")
	}
	badBuf := *valid
	badBuf.Scales = nil
	if _, err := UploadQ8_0PointerTable([]*GPUQ8_0Matrix{valid, &badBuf}); err == nil {
		t.Fatal("accepted Q8_0 matrix missing scale buffer")
	}
}

func TestGemvQ8_0ScatterByWorkPtrsRejectsBadInputs(t *testing.T) {
	validBuf := &Buffer{Ptr: 1, Size: 64}
	shortBuf := &Buffer{Ptr: 1, Size: 2}
	table := &GPUQ8_0PointerTable{QPtrs: validBuf, ScalePtrs: validBuf, InDim: 32, OutDim: 4, Count: 1}
	if err := GemvQ8_0ScatterByWorkPtrs(validBuf, validBuf, validBuf, validBuf, validBuf, 1, nil); err == nil {
		t.Fatal("accepted nil Q8 pointer table")
	}
	if err := GemvQ8_0ScatterByWorkPtrs(validBuf, shortBuf, validBuf, validBuf, validBuf, 1, table); err == nil {
		t.Fatal("accepted short x buffer")
	}
	badTable := *table
	badTable.ScalePtrs = nil
	if err := GemvQ8_0ScatterByWorkPtrs(validBuf, validBuf, validBuf, validBuf, validBuf, 1, &badTable); err == nil {
		t.Fatal("accepted table missing scale pointers")
	}
}

func TestUploadQ8_0MatrixRowsIntoReusesBuffers(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim := 32, 2
	mkRaw := func(scaleBase float32) []byte {
		raw := make([]byte, outDim*34)
		for r := 0; r < outDim; r++ {
			row := raw[r*34 : (r+1)*34]
			binary.LittleEndian.PutUint16(row[:2], half.F32ToF16(scaleBase+float32(r)*0.01))
			for i := 0; i < inDim; i++ {
				row[2+i] = byte(int8((r+i)%11 - 5))
			}
		}
		return raw
	}
	m, err := UploadQ8_0MatrixRows(mkRaw(0.05), inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	qPtr, sPtr := m.Q.Ptr, m.Scales.Ptr
	if err := UploadQ8_0MatrixRowsInto(m, mkRaw(0.15), inDim, outDim); err != nil {
		t.Fatal(err)
	}
	if m.Q.Ptr != qPtr || m.Scales.Ptr != sPtr {
		t.Fatalf("in-place upload changed pointers")
	}
	out := make([]float32, outDim)
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.02
	}
	if err := GemvQ8_0(out, x, m); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 && out[1] == 0 {
		t.Fatalf("unexpected zero output after in-place upload")
	}
}
