package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
)

const (
	experimentalQ4K8x8Rows    = 8
	experimentalQ4K8x8ActRows = 8
	experimentalQ4KBlockBytes = 144
	experimentalQ4KBlockElems = 256
)

type experimentalQ4KBlockX8 struct {
	d      [8]uint16
	dmin   [8]uint16
	scales [96]byte
	qs     [1024]byte
}

type experimentalQ8KBlockX4 struct {
	d     [4]float32
	qs    [1024]int8
	bsums [64]int16
}

// experimentalQ4K8x8Tile is a correctness-first port of ggml's current CPU
// repack+GEMM path for one canonical 8x8 Q4_K tile. It remains internal until
// a complete batched projection path consumes it.
type experimentalQ4K8x8Tile struct {
	k      int
	blocks []experimentalQ4KBlockX8
}

// newExperimentalQ4K8x8Tile repacks 8 canonical raw GGUF Q4_K rows into the
// current ggml-style 8x8 backend tile layout. It expects exactly 8 rows.
func newExperimentalQ4K8x8Tile(raw []byte, k int) (*experimentalQ4K8x8Tile, error) {
	if k <= 0 || k%experimentalQ4KBlockElems != 0 {
		return nil, fmt.Errorf("experimental q4k 8x8 tile: k=%d not multiple of %d", k, experimentalQ4KBlockElems)
	}
	rowBytes, err := TensorRawBytes(QuantQ4_K, k)
	if err != nil {
		return nil, fmt.Errorf("experimental q4k 8x8 tile: %w", err)
	}
	wantRaw := experimentalQ4K8x8Rows * rowBytes
	if len(raw) != wantRaw {
		return nil, fmt.Errorf("experimental q4k 8x8 tile: raw=%d want=%d", len(raw), wantRaw)
	}
	nb := k / experimentalQ4KBlockElems
	blocks := make([]experimentalQ4KBlockX8, nb)
	for bi := 0; bi < nb; bi++ {
		var rows [experimentalQ4K8x8Rows][]byte
		for r := 0; r < experimentalQ4K8x8Rows; r++ {
			start := r*rowBytes + bi*experimentalQ4KBlockBytes
			rows[r] = raw[start : start+experimentalQ4KBlockBytes]
		}
		blocks[bi] = repackExperimentalQ4KBlockX8(rows)
	}
	return &experimentalQ4K8x8Tile{k: k, blocks: blocks}, nil
}

// mulF32ActivationRows computes outputs for eight row-major F32 activation
// vectors. It returns [weight row][activation row], transposed from ggml's
// contiguous [activation row][weight row] mul_mat result for Go callers.
func (t *experimentalQ4K8x8Tile) mulF32ActivationRows(acts []float32) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("experimental q4k 8x8 tile: nil tile")
	}
	if len(acts) != experimentalQ4K8x8ActRows*t.k {
		return nil, fmt.Errorf("experimental q4k 8x8 tile: acts=%d want=%d", len(acts), experimentalQ4K8x8ActRows*t.k)
	}
	out := make([]float32, experimentalQ4K8x8Rows*experimentalQ4K8x8ActRows)
	for g := 0; g < experimentalQ4K8x8ActRows/4; g++ {
		q8 := quantizeExperimentalQ8K4x8(acts[g*4*t.k:(g+1)*4*t.k], t.k)
		batch := gemmExperimentalQ4K8x8Q8K(t.k, t.blocks, q8, 4, experimentalQ4K8x8Rows)
		for actRow := 0; actRow < 4; actRow++ {
			for weightRow := 0; weightRow < experimentalQ4K8x8Rows; weightRow++ {
				out[weightRow*experimentalQ4K8x8ActRows+g*4+actRow] = batch[actRow*experimentalQ4K8x8Rows+weightRow]
			}
		}
	}
	return out, nil
}

func repackExperimentalQ4KBlockX8(in [8][]byte) experimentalQ4KBlockX8 {
	const blockSizeInterleave = 8
	var out experimentalQ4KBlockX8
	for i := 0; i < 8; i++ {
		out.d[i] = binary.LittleEndian.Uint16(in[i][0:2])
		out.dmin[i] = binary.LittleEndian.Uint16(in[i][2:4])
	}
	end := experimentalQ4KBlockElems * 4 / blockSizeInterleave
	for i := 0; i < end; i++ {
		srcID := i % 8
		srcOffset := 16 + (i/8)*blockSizeInterleave
		dstOffset := i * blockSizeInterleave
		copy(out.qs[dstOffset:dstOffset+blockSizeInterleave], in[srcID][srcOffset:srcOffset+blockSizeInterleave])
	}
	var s, m [8]byte
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			s[j] = in[j][4+i] & 63
			m[j] = in[j][8+i] & 63
		}
		base := i * 12
		out.scales[base+0] = (s[0] & 63) + ((s[4] & 48) << 2)
		out.scales[base+1] = (s[1] & 63) + ((s[5] & 48) << 2)
		out.scales[base+2] = (s[2] & 63) + ((s[6] & 48) << 2)
		out.scales[base+3] = (s[3] & 63) + ((s[7] & 48) << 2)
		out.scales[base+4] = (m[0] & 63) + ((m[4] & 48) << 2)
		out.scales[base+5] = (m[1] & 63) + ((m[5] & 48) << 2)
		out.scales[base+6] = (m[2] & 63) + ((m[6] & 48) << 2)
		out.scales[base+7] = (m[3] & 63) + ((m[7] & 48) << 2)
		out.scales[base+8] = (s[4] & 15) + ((m[4] & 15) << 4)
		out.scales[base+9] = (s[5] & 15) + ((m[5] & 15) << 4)
		out.scales[base+10] = (s[6] & 15) + ((m[6] & 15) << 4)
		out.scales[base+11] = (s[7] & 15) + ((m[7] & 15) << 4)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			s[j] = ((in[j][4+i] & 192) >> 2) | (in[j][12+i] & 15)
			m[j] = ((in[j][8+i] & 192) >> 2) | ((in[j][12+i] & 240) >> 4)
		}
		base := 48 + i*12
		out.scales[base+0] = (s[0] & 63) + ((s[4] & 48) << 2)
		out.scales[base+1] = (s[1] & 63) + ((s[5] & 48) << 2)
		out.scales[base+2] = (s[2] & 63) + ((s[6] & 48) << 2)
		out.scales[base+3] = (s[3] & 63) + ((s[7] & 48) << 2)
		out.scales[base+4] = (m[0] & 63) + ((m[4] & 48) << 2)
		out.scales[base+5] = (m[1] & 63) + ((m[5] & 48) << 2)
		out.scales[base+6] = (m[2] & 63) + ((m[6] & 48) << 2)
		out.scales[base+7] = (m[3] & 63) + ((m[7] & 48) << 2)
		out.scales[base+8] = (s[4] & 15) + ((m[4] & 15) << 4)
		out.scales[base+9] = (s[5] & 15) + ((m[5] & 15) << 4)
		out.scales[base+10] = (s[6] & 15) + ((m[6] & 15) << 4)
		out.scales[base+11] = (s[7] & 15) + ((m[7] & 15) << 4)
	}
	return out
}

func quantizeExperimentalQ8K4x8(x []float32, k int) []experimentalQ8KBlockX4 {
	nb := k / experimentalQ4KBlockElems
	out := make([]experimentalQ8KBlockX4, nb)
	const blockSizeInterleave = 8
	var srcv [4][experimentalQ4KBlockElems]float32
	var iscale [4]float32
	for bi := 0; bi < nb; bi++ {
		for rowIter := 0; rowIter < 4; rowIter++ {
			amax := float32(0)
			maxv := float32(0)
			for j := 0; j < experimentalQ4KBlockElems; j++ {
				v := x[rowIter*k+bi*experimentalQ4KBlockElems+j]
				srcv[rowIter][j] = v
				av := float32(math.Abs(float64(v)))
				if amax < av {
					amax = av
					maxv = v
				}
			}
			if amax != 0 {
				iscale[rowIter] = -127 / maxv
				out[bi].d[rowIter] = 1 / iscale[rowIter]
			} else {
				iscale[rowIter] = 0
				out[bi].d[rowIter] = 0
			}
		}
		for j := 0; j < experimentalQ4KBlockElems/4; j++ {
			out[bi].bsums[j] = 0
		}
		for j := 0; j < experimentalQ4KBlockElems*4; j++ {
			srcOffset := (j / (4 * blockSizeInterleave)) * blockSizeInterleave
			srcID := (j % (4 * blockSizeInterleave)) / blockSizeInterleave
			srcOffset += j % blockSizeInterleave
			index := (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3)
			x0 := srcv[srcID][srcOffset] * iscale[srcID]
			q := nearestIntGGML(x0)
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			out[bi].qs[j] = int8(q)
			out[bi].bsums[index] += int16(q)
		}
	}
	return out
}

func gemmExperimentalQ4K8x8Q8K(n int, vx []experimentalQ4KBlockX8, vy []experimentalQ8KBlockX4, nr, nc int) []float32 {
	const (
		ncolsInterleaved = 8
		blocklen         = 8
	)
	const kmask1 uint32 = 0x3f3f3f3f
	const kmask2 uint32 = 0x0f0f0f0f
	const kmask3 uint32 = 0x03030303
	nb := n / experimentalQ4KBlockElems
	out := make([]float32, nr*nc)
	var sumf [4][8]float32
	var sumMinf [4][8]float32
	var utmp [32]uint32
	var tmpBytes [128]byte
	for y := 0; y < nr/4; y++ {
		aPtr := vy[y*nb : (y+1)*nb]
		for x := 0; x < nc/ncolsInterleaved; x++ {
			bPtr := vx[x*nb : (x+1)*nb]
			for m := 0; m < 4; m++ {
				for j := 0; j < ncolsInterleaved; j++ {
					sumf[m][j] = 0
					sumMinf[m][j] = 0
				}
			}
			for l := 0; l < nb; l++ {
				for sb := 0; sb < 8; sb++ {
					base := sb * 12
					u0 := binary.LittleEndian.Uint32(bPtr[l].scales[base : base+4])
					u1 := binary.LittleEndian.Uint32(bPtr[l].scales[base+4 : base+8])
					u2 := binary.LittleEndian.Uint32(bPtr[l].scales[base+8 : base+12])
					utmp[sb*4+0] = u0 & kmask1
					utmp[sb*4+1] = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4)
					utmp[sb*4+2] = u1 & kmask1
					utmp[sb*4+3] = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4)
					for w := 0; w < 4; w++ {
						binary.LittleEndian.PutUint32(tmpBytes[sb*16+w*4:sb*16+w*4+4], utmp[sb*4+w])
					}
				}
				for k := 0; k < experimentalQ4KBlockElems/(2*blocklen); k++ {
					scales0 := tmpBytes[(k/4)*32:]
					scales1 := tmpBytes[(k/4)*32+16:]
					for m := 0; m < 4; m++ {
						for j := 0; j < ncolsInterleaved; j++ {
							sumi := 0
							for i := 0; i < blocklen; i++ {
								qv := bPtr[l].qs[k*ncolsInterleaved*blocklen+j*blocklen+i]
								v0 := int(int8(qv & 0x0f))
								v1 := int(int8(qv >> 4))
								a0 := int(aPtr[l].qs[(k>>2)*256+(k%4)*4*blocklen+m*blocklen+i])
								a1 := int(aPtr[l].qs[(k>>2)*256+(k%4)*4*blocklen+m*blocklen+i+128])
								sumi += v0*a0*int(scales0[j]) + v1*a1*int(scales1[j])
							}
							sumf[m][j] += float32(sumi) * half.F16ToF32(bPtr[l].d[j]) * aPtr[l].d[m]
						}
					}
				}
				for sb := 0; sb < 8; sb++ {
					mins := tmpBytes[8+sb*16:]
					for m := 0; m < 4; m++ {
						base := sb*8 + m*4 - (sb%2)*6
						bsum := int(aPtr[l].bsums[base]) + int(aPtr[l].bsums[base+1])
						for j := 0; j < ncolsInterleaved; j++ {
							sumMinf[m][j] += float32(int(mins[j])*bsum) * half.F16ToF32(bPtr[l].dmin[j]) * aPtr[l].d[m]
						}
					}
				}
			}
			for m := 0; m < 4; m++ {
				for j := 0; j < ncolsInterleaved; j++ {
					out[(y*4+m)*nc+x*ncolsInterleaved+j] = sumf[m][j] - sumMinf[m][j]
				}
			}
		}
	}
	return out
}
