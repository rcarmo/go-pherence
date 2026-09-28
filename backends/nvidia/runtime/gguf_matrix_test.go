package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGPUGGUFMatrixDispatchesAdmittedKTypes(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device available but PTX runtime not ready")
		}
		t.Skip("CUDA unavailable")
	}
	cases := []struct {
		qt    gguf.QuantType
		block int
	}{{gguf.QuantQ4_K, 144}, {gguf.QuantQ5_K, 176}, {gguf.QuantQ6_K, 210}}
	for _, tc := range cases {
		t.Run(tc.qt.String(), func(t *testing.T) {
			raw := make([]byte, tc.block*2)
			for b := 0; b < 2; b++ {
				blk := raw[b*tc.block : (b+1)*tc.block]
				switch tc.qt {
				case gguf.QuantQ4_K, gguf.QuantQ5_K:
					binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.02))
					binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.003))
				case gguf.QuantQ6_K:
					for i := 192; i < 208; i++ {
						blk[i] = 1
					}
					binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(0.02))
				}
			}
			m := &gguf.QuantMatrix{Name: "fixture", QType: tc.qt, Raw: raw, InDim: 512, OutDim: 1}
			gm, err := UploadGGUFMatrix(m)
			if err != nil {
				t.Fatal(err)
			}
			defer gm.Free()
			x, _ := Malloc(512)
			out, _ := Malloc(1)
			defer x.Free()
			defer out.Free()
			if err := x.Upload(make([]float32, 512)); err != nil {
				t.Fatal(err)
			}
			if err := gm.ProjectBatchToBuffer(out, x, 1); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestProjectQ4RawPairMatchesSeparateProjections(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const inDim, outDim, batch = 256, 256, 24
	makeMatrix := func(offset int) *gguf.QuantMatrix {
		raw := make([]byte, outDim*144)
		for r := 0; r < outDim; r++ {
			block := raw[r*144 : (r+1)*144]
			binary.LittleEndian.PutUint16(block[:2], half.F32ToF16(.02))
			binary.LittleEndian.PutUint16(block[2:4], half.F32ToF16(.003))
			for i := 0; i < 12; i++ {
				block[4+i] = byte(1 + (i+r+offset)%12)
			}
			for i := 0; i < 128; i++ {
				block[16+i] = byte((i*7 + r*11 + offset) & 0xff)
			}
		}
		return &gguf.QuantMatrix{Name: "fixture", QType: gguf.QuantQ4_K, Raw: raw, InDim: inDim, OutDim: outDim}
	}
	a, err := UploadGGUFMatrix(makeMatrix(0))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := UploadGGUFMatrix(makeMatrix(3))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	x, _ := Malloc(batch * inDim)
	wantA, _ := Malloc(batch * outDim)
	wantB, _ := Malloc(batch * outDim)
	gotA, _ := Malloc(batch * outDim)
	gotB, _ := Malloc(batch * outDim)
	defer x.Free()
	defer wantA.Free()
	defer wantB.Free()
	defer gotA.Free()
	defer gotB.Free()
	host := make([]float32, batch*inDim)
	for i := range host {
		host[i] = float32((i%31)-15) / 17
	}
	if err := x.Upload(host); err != nil {
		t.Fatal(err)
	}
	if err := a.ProjectBatchToBuffer(wantA, x, batch); err != nil {
		t.Fatal(err)
	}
	if err := b.ProjectBatchToBuffer(wantB, x, batch); err != nil {
		t.Fatal(err)
	}
	if err := ProjectQ4PairToBuffers(gotA, gotB, x, batch, a, b); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]*Buffer{{wantA, gotA}, {wantB, gotB}} {
		want, got := make([]float32, batch*outDim), make([]float32, batch*outDim)
		if err := pair[0].Download(want); err != nil {
			t.Fatal(err)
		}
		if err := pair[1].Download(got); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff != 0 {
				t.Fatalf("index=%d got=%g want=%g diff=%g", i, got[i], want[i], diff)
			}
		}
	}
}

func TestGPUGGUFMatrixRejectsUnsupportedType(t *testing.T) {
	if _, err := UploadGGUFMatrix(&gguf.QuantMatrix{Name: "bad", QType: gguf.QuantF32, InDim: 2, OutDim: 2, Raw: make([]byte, 16)}); err == nil {
		t.Fatal("accepted F32 as admitted quantized matrix")
	}
}

func TestSelectedBatchMatchesSingleRow(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const width, vocab, rows = 512, 7, 3
	raw := make([]byte, vocab*(width/256)*176)
	for r := 0; r < vocab*(width/256); r++ {
		b := raw[r*176 : (r+1)*176]
		for i := range b {
			b[i] = byte((r*13 + i*7) % 256)
		}
		binary.LittleEndian.PutUint16(b[:2], half.F32ToF16(.013))
		binary.LittleEndian.PutUint16(b[2:4], half.F32ToF16(.017))
	}
	m, err := UploadGGUFMatrix(&gguf.QuantMatrix{Name: "selected fixture", QType: gguf.QuantQ5_K, Raw: raw, InDim: width, OutDim: vocab})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x, err := Malloc(rows * width)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	host := make([]float32, rows*width)
	for i := range host {
		host[i] = float32(math.Sin(float64(i) * .7))
	}
	if err = x.Upload(host); err != nil {
		t.Fatal(err)
	}
	ids := [][]int{{6, 0, 2}, {1}, {3, 2}}
	got, err := m.ProjectSelectedBatch(x, ids)
	if err != nil {
		t.Fatal(err)
	}
	for i, list := range ids {
		b, err := Malloc(len(list))
		if err != nil {
			t.Fatal(err)
		}
		defer b.Free()
		out, err := Malloc(len(list))
		if err != nil {
			t.Fatal(err)
		}
		defer out.Free()
		selected := make([]uint32, len(list))
		for j, id := range list {
			selected[j] = uint32(id)
		}
		if err = b.UploadUint32(selected); err != nil {
			t.Fatal(err)
		}
		view := &Buffer{Ptr: x.Ptr + CUdeviceptr(i*width*4), Size: width * 4}
		if err = m.ProjectSelectedRows(out, view, b, len(list)); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, len(list))
		if err = out.Download(want); err != nil {
			t.Fatal(err)
		}
		for j, v := range want {
			if got[i][j] != v {
				t.Fatalf("[%d][%d] got=%g want=%g", i, j, got[i][j], v)
			}
		}
	}
	for _, bad := range [][][]int{nil, {{-1}}, {{vocab}}, {{}}} {
		if _, err = m.ProjectSelectedBatch(x, bad); err == nil {
			t.Fatal("bad IDs accepted")
		}
	}
}
