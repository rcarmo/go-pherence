package nvidia

import (
	"encoding/binary"
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

func TestGPUGGUFMatrixRejectsUnsupportedType(t *testing.T) {
	if _, err := UploadGGUFMatrix(&gguf.QuantMatrix{Name: "bad", QType: gguf.QuantF32, InDim: 2, OutDim: 2, Raw: make([]byte, 16)}); err == nil {
		t.Fatal("accepted F32 as admitted quantized matrix")
	}
}
