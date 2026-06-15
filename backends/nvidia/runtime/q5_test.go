package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGemvQ5_0BatchMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, batch := 64, 5, 3
	rowBytes := (inDim / 32) * 22
	raw := make([]byte, outDim*rowBytes)
	for r := 0; r < outDim; r++ {
		for b := 0; b < inDim/32; b++ {
			blk := raw[r*rowBytes+b*22:]
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.025+float32(r+b)*0.003))
			var high uint32
			for i := 0; i < 32; i++ {
				q := (i*3 + r*5 + b*7) % 32
				if q >= 16 {
					high |= 1 << uint(i)
				}
				if i < 16 {
					blk[6+i] = (blk[6+i] & 0xF0) | byte(q&0x0F)
				} else {
					blk[6+i-16] = (blk[6+i-16] & 0x0F) | byte((q&0x0F)<<4)
				}
			}
			binary.LittleEndian.PutUint32(blk[2:6], high)
		}
	}
	m, err := UploadQ5_0MatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.021
	}
	got := make([]float32, batch*outDim)
	if err := GemvQ5_0Batch(got, x, batch, m); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < outDim; r++ {
		deq := dequantQ5_0Test(raw[r*rowBytes:(r+1)*rowBytes], inDim)
		for b := 0; b < batch; b++ {
			var want float32
			xrow := x[b*inDim : (b+1)*inDim]
			for i := range deq {
				want += deq[i] * xrow[i]
			}
			if math.Abs(float64(got[b*outDim+r]-want)) > 1e-4 {
				t.Fatalf("batch=%d row=%d got=%g want=%g all=%v", b, r, got[b*outDim+r], want, got)
			}
		}
	}
}

func TestGemvQ5_0ScatterByWork(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, active, workLen := 64, 5, 2, 4
	rowBytes := (inDim / 32) * 22
	raw := make([]byte, active*outDim*rowBytes)
	for r := 0; r < active*outDim; r++ {
		for b := 0; b < inDim/32; b++ {
			blk := raw[r*rowBytes+b*22:]
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.02+float32(r+b)*0.0025))
			var high uint32
			for i := 0; i < 32; i++ {
				q := (i*5 + r*3 + b*11) % 32
				if q >= 16 {
					high |= 1 << uint(i)
				}
				if i < 16 {
					blk[6+i] = (blk[6+i] & 0xF0) | byte(q&0x0F)
				} else {
					blk[6+i-16] = (blk[6+i-16] & 0x0F) | byte((q&0x0F)<<4)
				}
			}
			binary.LittleEndian.PutUint32(blk[2:6], high)
		}
	}
	m, err := UploadQ5_0MatrixRows(raw, inDim, active*outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.023
	}
	workActive := []uint32{0, 1, 0, 1}
	workPos := []uint32{0, 1, 1, 0}
	workWeight := []float32{0.5, 0.25, 0.75, 1.25}
	xBuf, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xBuf.Free()
	waBuf, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer waBuf.Free()
	wpBuf, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wpBuf.Free()
	wwBuf, err := Malloc(workLen)
	if err != nil {
		t.Fatal(err)
	}
	defer wwBuf.Free()
	dstBuf, err := Malloc(2 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dstBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := waBuf.UploadUint32(workActive); err != nil {
		t.Fatal(err)
	}
	if err := wpBuf.UploadUint32(workPos); err != nil {
		t.Fatal(err)
	}
	if err := wwBuf.Upload(workWeight); err != nil {
		t.Fatal(err)
	}
	if err := ZeroFloat32Buffer(dstBuf, 2*outDim); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ5_0ScatterByWork(dstBuf, xBuf, waBuf, wpBuf, wwBuf, workLen, active, m); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 2*outDim)
	if err := dstBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 2*outDim)
	for w := 0; w < workLen; w++ {
		expert := int(workActive[w])
		pos := int(workPos[w])
		xrow := x[w*inDim : (w+1)*inDim]
		for r := 0; r < outDim; r++ {
			deq := dequantQ5_0Test(raw[(expert*outDim+r)*rowBytes:(expert*outDim+r+1)*rowBytes], inDim)
			var dot float32
			for i := range deq {
				dot += deq[i] * xrow[i]
			}
			want[pos*outDim+r] += workWeight[w] * dot
		}
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("i=%d got=%g want=%g all=%v", i, got[i], want[i], got)
		}
	}
}

func TestGemvQ5_0ScatterByWorkPtrs(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, active, workLen := 64, 5, 2, 4
	rowBytes := (inDim / 32) * 22
	raw := make([]byte, active*outDim*rowBytes)
	for r := 0; r < active*outDim; r++ {
		for b := 0; b < inDim/32; b++ {
			blk := raw[r*rowBytes+b*22:]
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.02+float32(r+b)*0.0025))
			var high uint32
			for i := 0; i < 32; i++ {
				q := (i*5 + r*3 + b*11) % 32
				if q >= 16 {
					high |= 1 << uint(i)
				}
				if i < 16 {
					blk[6+i] = (blk[6+i] & 0xF0) | byte(q&0x0F)
				} else {
					blk[6+i-16] = (blk[6+i-16] & 0x0F) | byte((q&0x0F)<<4)
				}
			}
			binary.LittleEndian.PutUint32(blk[2:6], high)
		}
	}
	packed, err := UploadQ5_0MatrixRows(raw, inDim, active*outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer packed.Free()
	mats := make([]*GPUQ5_0Matrix, active)
	for a := 0; a < active; a++ {
		m, err := UploadQ5_0MatrixRows(raw[a*outDim*rowBytes:(a+1)*outDim*rowBytes], inDim, outDim)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Free()
		mats[a] = m
	}
	table, err := UploadQ5_0PointerTable(mats)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Free()
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.023
	}
	workActive := []uint32{0, 1, 0, 1}
	workPos := []uint32{0, 1, 1, 0}
	workWeight := []float32{0.5, 0.25, 0.75, 1.25}
	xBuf, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xBuf.Free()
	waBuf, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer waBuf.Free()
	wpBuf, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer wpBuf.Free()
	wwBuf, err := Malloc(workLen)
	if err != nil {
		t.Fatal(err)
	}
	defer wwBuf.Free()
	wantBuf, err := Malloc(2 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer wantBuf.Free()
	gotBuf, err := Malloc(2 * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer gotBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := waBuf.UploadUint32(workActive); err != nil {
		t.Fatal(err)
	}
	if err := wpBuf.UploadUint32(workPos); err != nil {
		t.Fatal(err)
	}
	if err := wwBuf.Upload(workWeight); err != nil {
		t.Fatal(err)
	}
	if err := ZeroFloat32Buffer(wantBuf, 2*outDim); err != nil {
		t.Fatal(err)
	}
	if err := ZeroFloat32Buffer(gotBuf, 2*outDim); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ5_0ScatterByWork(wantBuf, xBuf, waBuf, wpBuf, wwBuf, workLen, active, packed); err != nil {
		t.Fatal(err)
	}
	if err := GemvQ5_0ScatterByWorkPtrs(gotBuf, xBuf, waBuf, wpBuf, wwBuf, workLen, table); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 2*outDim)
	got := make([]float32, 2*outDim)
	if err := wantBuf.Download(want); err != nil {
		t.Fatal(err)
	}
	if err := gotBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("i=%d got=%g want=%g", i, got[i], want[i])
		}
	}
}

func TestGemvQ5_0ScatterByWorkPtrsRejectsBadInputs(t *testing.T) {
	validBuf := &Buffer{Ptr: 1, Size: 64}
	shortBuf := &Buffer{Ptr: 1, Size: 2}
	table := &GPUQ5_0PointerTable{QPtrs: validBuf, HighPtrs: validBuf, ScalePtrs: validBuf, InDim: 32, OutDim: 4, Count: 1}
	if err := GemvQ5_0ScatterByWorkPtrs(validBuf, validBuf, validBuf, validBuf, validBuf, 1, nil); err == nil {
		t.Fatal("accepted nil Q5 pointer table")
	}
	if err := GemvQ5_0ScatterByWorkPtrs(validBuf, shortBuf, validBuf, validBuf, validBuf, 1, table); err == nil {
		t.Fatal("accepted short x buffer")
	}
	badTable := *table
	badTable.HighPtrs = nil
	if err := GemvQ5_0ScatterByWorkPtrs(validBuf, validBuf, validBuf, validBuf, validBuf, 1, &badTable); err == nil {
		t.Fatal("accepted table missing high-bit pointers")
	}
}

func dequantQ5_0Test(raw []byte, n int) []float32 {
	out := make([]float32, n)
	for b := 0; b < n/32; b++ {
		blk := raw[b*22:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
		high := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		for i := 0; i < 16; i++ {
			q0 := int(qs[i] & 0x0F)
			q1 := int(qs[i] >> 4)
			if high&(1<<uint(i)) != 0 {
				q0 |= 16
			}
			if high&(1<<uint(i+16)) != 0 {
				q1 |= 16
			}
			out[b*32+i] = d * float32(q0-16)
			out[b*32+i+16] = d * float32(q1-16)
		}
	}
	return out
}
