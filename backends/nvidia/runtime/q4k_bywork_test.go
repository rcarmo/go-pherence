package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGateUpGELUQ4KByWorkToBuffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, inter, active := 256, 2, 2
	outDim := inter * 2 * active
	raw := make([]byte, outDim*144)
	for r := 0; r < outDim; r++ {
		blk := raw[r*144 : (r+1)*144]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.035+float32(r)*0.003))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.008))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%11)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*5 + r) & 0xff)
		}
	}
	m, err := UploadQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	workLen := 3
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.02
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	we, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer we.Free()
	out, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := we.UploadUint32([]uint32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := GateUpGELUQ4KByWorkToBuffer(out, xb, we, workLen, inter, active, m); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, workLen*inter)
	if err := out.Download(got); err != nil {
		t.Fatal(err)
	}
	for w := 0; w < workLen; w++ {
		exp := []int{0, 1, 0}[w]
		rowOffset := exp * inter * 2
		subRaw := raw[rowOffset*144 : (rowOffset+inter*2)*144]
		sub, err := UploadQ4KMatrixRows(subRaw, inDim, inter*2)
		if err != nil {
			t.Fatal(err)
		}
		buf, err := Malloc(inter)
		if err != nil {
			t.Fatal(err)
		}
		xone, err := Malloc(inDim)
		if err != nil {
			t.Fatal(err)
		}
		if err := xone.Upload(x[w*inDim : (w+1)*inDim]); err != nil {
			t.Fatal(err)
		}
		if err := GateUpGELUQ4KBatchToBuffer(buf, xone, 1, inter, sub); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, inter)
		if err := buf.Download(want); err != nil {
			t.Fatal(err)
		}
		sub.Free()
		buf.Free()
		xone.Free()
		for i := 0; i < inter; i++ {
			if math.Abs(float64(got[w*inter+i]-want[i])) > 1e-4 {
				t.Fatalf("work=%d i=%d got=%g want=%g", w, i, got[w*inter+i], want[i])
			}
		}
	}
}

func TestGateUpGELUQ4KByWorkPtrsToBuffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, inter, active := 256, 2, 2
	outDim := inter * 2 * active
	raw := make([]byte, outDim*144)
	for r := 0; r < outDim; r++ {
		blk := raw[r*144 : (r+1)*144]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.035+float32(r)*0.003))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.008))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%11)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*5 + r) & 0xff)
		}
	}
	packed, err := UploadQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer packed.Free()
	mats := make([]*GPUQ4KMatrix, active)
	for a := 0; a < active; a++ {
		m, err := UploadQ4KMatrixRows(raw[a*inter*2*144:(a+1)*inter*2*144], inDim, inter*2)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Free()
		mats[a] = m
	}
	table, err := UploadQ4KPointerTable(mats)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Free()
	workLen := 3
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.02
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	we, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer we.Free()
	wantBuf, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer wantBuf.Free()
	gotBuf, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer gotBuf.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := we.UploadUint32([]uint32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := GateUpGELUQ4KByWorkToBuffer(wantBuf, xb, we, workLen, inter, active, packed); err != nil {
		t.Fatal(err)
	}
	if err := GateUpGELUQ4KByWorkPtrsToBuffer(gotBuf, xb, we, workLen, inter, table); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, workLen*inter)
	got := make([]float32, workLen*inter)
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

func TestGateUpQ4KByWorkPtrsToBuffers(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, inter, active := 256, 3, 2
	raw := make([]byte, active*inter*2*144)
	for r := 0; r < active*inter*2; r++ {
		blk := raw[r*144 : (r+1)*144]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(r)*0.002))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.006))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%13)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*7 + r*3) & 0xff)
		}
	}
	mats := make([]*GPUQ4KMatrix, active)
	for a := 0; a < active; a++ {
		m, err := UploadQ4KMatrixRows(raw[a*inter*2*144:(a+1)*inter*2*144], inDim, inter*2)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Free()
		mats[a] = m
	}
	table, err := UploadQ4KPointerTable(mats)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Free()
	workLen := 3
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%19)-9) * 0.017
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	we, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer we.Free()
	gateBuf, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer gateBuf.Free()
	upBuf, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer upBuf.Free()
	workExperts := []uint32{0, 1, 0}
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := we.UploadUint32(workExperts); err != nil {
		t.Fatal(err)
	}
	if err := GateUpQ4KByWorkPtrsToBuffers(gateBuf, upBuf, xb, we, workLen, inter, table); err != nil {
		t.Fatal(err)
	}
	gotGate := make([]float32, workLen*inter)
	gotUp := make([]float32, workLen*inter)
	if err := gateBuf.Download(gotGate); err != nil {
		t.Fatal(err)
	}
	if err := upBuf.Download(gotUp); err != nil {
		t.Fatal(err)
	}
	for w := 0; w < workLen; w++ {
		expert := int(workExperts[w])
		xrow := x[w*inDim : (w+1)*inDim]
		for r := 0; r < inter; r++ {
			gateRow := raw[(expert*inter*2+r)*144 : (expert*inter*2+r+1)*144]
			upRow := raw[(expert*inter*2+inter+r)*144 : (expert*inter*2+inter+r+1)*144]
			gateDeq := dequantQ4KTest(gateRow, inDim)
			upDeq := dequantQ4KTest(upRow, inDim)
			var wantGate, wantUp float32
			for i := range xrow {
				wantGate += gateDeq[i] * xrow[i]
				wantUp += upDeq[i] * xrow[i]
			}
			idx := w*inter + r
			if math.Abs(float64(gotGate[idx]-wantGate)) > 1e-4 || math.Abs(float64(gotUp[idx]-wantUp)) > 1e-4 {
				t.Fatalf("work=%d row=%d gate=%g/%g up=%g/%g", w, r, gotGate[idx], wantGate, gotUp[idx], wantUp)
			}
		}
	}
}

func TestUploadQ4KMatrixRowsRejectsInvalidInput(t *testing.T) {
	if _, err := UploadQ4KMatrixRows(nil, 255, 1); err == nil {
		t.Fatal("accepted non-256-aligned Q4_K input dimension")
	}
	if _, err := UploadQ4KMatrixRows(make([]byte, 143), 256, 1); err == nil {
		t.Fatal("accepted short Q4_K raw row")
	}
	if _, err := UploadQ4KMatrixRows(make([]byte, 144), 256, 0); err == nil {
		t.Fatal("accepted zero Q4_K output dimension")
	}
}

func TestUploadQ4KPointerTableRejectsInvalidMatrices(t *testing.T) {
	validBuf := &Buffer{Ptr: 1, Size: 64}
	valid := &GPUQ4KMatrix{Q: validBuf, Scales: validBuf, Mins: validBuf, InDim: 256, OutDim: 4}
	if _, err := UploadQ4KPointerTable(nil); err == nil {
		t.Fatal("accepted empty Q4_K pointer table")
	}
	if _, err := UploadQ4KPointerTable([]*GPUQ4KMatrix{valid, nil}); err == nil {
		t.Fatal("accepted nil Q4_K matrix")
	}
	badDims := *valid
	badDims.OutDim = 8
	if _, err := UploadQ4KPointerTable([]*GPUQ4KMatrix{valid, &badDims}); err == nil {
		t.Fatal("accepted mismatched Q4_K matrix dimensions")
	}
	badBuf := *valid
	badBuf.Mins = nil
	if _, err := UploadQ4KPointerTable([]*GPUQ4KMatrix{valid, &badBuf}); err == nil {
		t.Fatal("accepted Q4_K matrix missing min buffer")
	}
}

func TestGateUpQ4KByWorkPtrsToBuffersRejectsBadInputs(t *testing.T) {
	validBuf := &Buffer{Ptr: 1, Size: 4096}
	shortBuf := &Buffer{Ptr: 1, Size: 2}
	table := &GPUQ4KPointerTable{QPtrs: validBuf, ScalePtrs: validBuf, MinPtrs: validBuf, InDim: 256, OutDim: 4, Count: 1}
	if err := GateUpQ4KByWorkPtrsToBuffers(validBuf, validBuf, validBuf, validBuf, 1, 2, nil); err == nil {
		t.Fatal("accepted nil Q4_K pointer table")
	}
	if err := GateUpQ4KByWorkPtrsToBuffers(validBuf, validBuf, shortBuf, validBuf, 1, 2, table); err == nil {
		t.Fatal("accepted short x buffer")
	}
	badTable := *table
	badTable.MinPtrs = nil
	if err := GateUpQ4KByWorkPtrsToBuffers(validBuf, validBuf, validBuf, validBuf, 1, 2, &badTable); err == nil {
		t.Fatal("accepted table missing min pointers")
	}
}

func TestUploadQ4KMatrixRowsIntoReusesBuffers(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim := 256, 2
	mkRaw := func(scaleBase float32) []byte {
		raw := make([]byte, outDim*144)
		for r := 0; r < outDim; r++ {
			blk := raw[r*144 : (r+1)*144]
			binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(scaleBase+float32(r)*0.003))
			binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.008))
			for i := 0; i < 12; i++ {
				blk[4+i] = byte(1 + (i+r)%11)
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte((i*5 + r) & 0xff)
			}
		}
		return raw
	}
	m, err := UploadQ4KMatrixRows(mkRaw(0.035), inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	qPtr, sPtr, minPtr := m.Q.Ptr, m.Scales.Ptr, m.Mins.Ptr
	if err := UploadQ4KMatrixRowsInto(m, mkRaw(0.055), inDim, outDim); err != nil {
		t.Fatal(err)
	}
	if m.Q.Ptr != qPtr || m.Scales.Ptr != sPtr || m.Mins.Ptr != minPtr {
		t.Fatalf("in-place upload changed pointers")
	}
	out := make([]float32, outDim)
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.02
	}
	if err := GemvQ4K(out, x, m); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 && out[1] == 0 {
		t.Fatalf("unexpected zero output after in-place upload")
	}
}
