package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGateUpQ4KByWorkPtrsCompactScaleParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, inter, active := 512, 3, 2
	blocks := inDim / 256
	rowBytes := blocks * 144
	raw := make([]byte, active*inter*2*rowBytes)
	for r := 0; r < active*inter*2; r++ {
		for b := 0; b < blocks; b++ {
			blk := raw[r*rowBytes+b*144 : r*rowBytes+(b+1)*144]
			binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(r+b)*0.0025))
			binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.006+float32((r+b)%3)*0.001))
			for i := 0; i < 12; i++ {
				blk[4+i] = byte((0x40 + i*17 + r*29 + b*31) & 0xff)
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte((i*7 + r*3 + b*11) & 0xff)
			}
		}
	}
	mats := make([]*GPUQ4KMatrix, active)
	matsRaw := make([]*GPUQ4KMatrixRaw, active)
	for a := 0; a < active; a++ {
		sub := raw[a*inter*2*rowBytes : (a+1)*inter*2*rowBytes]
		m, err := UploadQ4KMatrixRows(sub, inDim, inter*2)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Free()
		mats[a] = m
		mr, err := UploadQ4KMatrixRowsRaw(sub, inDim, inter*2)
		if err != nil {
			t.Fatal(err)
		}
		defer mr.Free()
		matsRaw[a] = mr
	}
	table, err := UploadQ4KPointerTable(mats)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Free()
	tableRaw, err := UploadQ4KPointerTableRaw(matsRaw)
	if err != nil {
		t.Fatal(err)
	}
	defer tableRaw.Free()
	workLen := 4
	x := make([]float32, workLen*inDim)
	for i := range x {
		x[i] = float32((i%19)-9) * 0.017
	}
	xb, err := Malloc(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	we, err := MallocBytes(workLen * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer we.Free()
	if err := we.UploadUint32([]uint32{0, 1, 0, 1}); err != nil {
		t.Fatal(err)
	}
	g0, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer g0.Free()
	u0, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer u0.Free()
	g1, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer g1.Free()
	u1, err := Malloc(workLen * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer u1.Free()
	if err := GateUpQ4KByWorkPtrsToBuffers(g0, u0, xb, we, workLen, inter, table); err != nil {
		t.Fatal(err)
	}
	if err := GateUpQ4KByWorkPtrsRawToBuffers(g1, u1, xb, we, workLen, inter, tableRaw); err != nil {
		t.Fatal(err)
	}
	wantG, wantU := make([]float32, workLen*inter), make([]float32, workLen*inter)
	gotG, gotU := make([]float32, workLen*inter), make([]float32, workLen*inter)
	if err := g0.Download(wantG); err != nil {
		t.Fatal(err)
	}
	if err := u0.Download(wantU); err != nil {
		t.Fatal(err)
	}
	if err := g1.Download(gotG); err != nil {
		t.Fatal(err)
	}
	if err := u1.Download(gotU); err != nil {
		t.Fatal(err)
	}
	for i := range wantG {
		if math.Abs(float64(gotG[i]-wantG[i])) > 1e-5 || math.Abs(float64(gotU[i]-wantU[i])) > 1e-5 {
			t.Fatalf("i=%d gate got/want=%g/%g up got/want=%g/%g", i, gotG[i], wantG[i], gotU[i], wantU[i])
		}
	}
}
