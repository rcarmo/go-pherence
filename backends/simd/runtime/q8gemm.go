package simd

import (
	"encoding/binary"
	"math"
	"runtime"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
	"golang.org/x/sys/cpu"
)

const (
	q8_0BlockElems = 32
	q8_0BlockBytes = 34
)

// SgemmNTQ8_0To computes C = A * W^T for contiguous row-major A[m,k],
// Q8_0 weights W[n,k], and contiguous row-major C[m,n].
//
// Each Q8_0 block stores one FP16 scale followed by 32 signed int8 values.
// The function validates dimensions, slice footprints, output/input overlap,
// and that every raw FP16 scale is finite before entering the hot loop. The
// multiply kernel itself therefore assumes validated weights, which matches the
// loader contract for GGUF Q8_0 tensors. On success C is fully overwritten and
// no allocations occur.
func SgemmNTQ8_0To(c, a []float32, w []byte, m, n, k int) bool {
	blocks, rowBytes, needA, needC, needW, ok := validSgemmNTQ8_0Args(c, a, w, m, n, k)
	if !ok {
		return false
	}
	c = c[:needC]
	a = a[:needA]
	w = w[:needW]
	if !float32SlicesDisjoint(c, a) || !float32ByteSlicesDisjoint(c, w) {
		return false
	}
	if !q8_0FiniteScales(w, blocks, n) {
		return false
	}

	if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		return sgemmNTQ8_0ToSIMD(c, a, w, m, n, k, blocks, rowBytes)
	}
	return sgemmNTQ8_0ToScalar(c, a, w, m, n, k, blocks, rowBytes)
}

func sgemmNTQ8_0ToSIMD(c, a []float32, w []byte, m, n, k, blocks, rowBytes int) bool {
	i := 0
	for ; i+4 <= m; i += 4 {
		aRows := a[i*k:]
		c0 := (i + 0) * n
		c1 := (i + 1) * n
		c2 := (i + 2) * n
		c3 := (i + 3) * n
		for j := 0; j < n; j++ {
			row := w[j*rowBytes : (j+1)*rowBytes]
			var s0, s1, s2, s3 float32
			for b := 0; b < blocks; b++ {
				blk := row[b*q8_0BlockBytes:]
				d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
				v0, v1, v2, v3, ok := DotI8F32x4(blk[2:2+q8_0BlockElems], aRows[b*q8_0BlockElems:], k)
				if !ok {
					return false
				}
				s0 += d * v0
				s1 += d * v1
				s2 += d * v2
				s3 += d * v3
			}
			c[c0+j] = s0
			c[c1+j] = s1
			c[c2+j] = s2
			c[c3+j] = s3
		}
	}
	for ; i < m; i++ {
		arow := a[i*k : (i+1)*k]
		crow := c[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			row := w[j*rowBytes : (j+1)*rowBytes]
			var sum float32
			for b := 0; b < blocks; b++ {
				blk := row[b*q8_0BlockBytes:]
				d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
				v, ok := DotI8F32(blk[2:2+q8_0BlockElems], arow[b*q8_0BlockElems:])
				if !ok {
					return false
				}
				sum += d * v
			}
			crow[j] = sum
		}
	}
	return true
}

func sgemmNTQ8_0ToScalar(c, a []float32, w []byte, m, n, k, blocks, rowBytes int) bool {
	i := 0
	for ; i+4 <= m; i += 4 {
		aRows := a[i*k:]
		c0 := (i + 0) * n
		c1 := (i + 1) * n
		c2 := (i + 2) * n
		c3 := (i + 3) * n
		for j := 0; j < n; j++ {
			row := w[j*rowBytes : (j+1)*rowBytes]
			var s0, s1, s2, s3 float32
			for b := 0; b < blocks; b++ {
				blk := row[b*q8_0BlockBytes:]
				d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
				v0, v1, v2, v3 := dotI8F32x4Scalar(blk[2:2+q8_0BlockElems], aRows[b*q8_0BlockElems:], k)
				s0 += d * v0
				s1 += d * v1
				s2 += d * v2
				s3 += d * v3
			}
			c[c0+j] = s0
			c[c1+j] = s1
			c[c2+j] = s2
			c[c3+j] = s3
		}
	}
	for ; i < m; i++ {
		arow := a[i*k : (i+1)*k]
		crow := c[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			row := w[j*rowBytes : (j+1)*rowBytes]
			var sum float32
			for b := 0; b < blocks; b++ {
				blk := row[b*q8_0BlockBytes:]
				d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
				sum += d * dotI8F32Scalar(blk[2:2+q8_0BlockElems], arow[b*q8_0BlockElems:])
			}
			crow[j] = sum
		}
	}
	return true
}

func validSgemmNTQ8_0Args(c, a []float32, w []byte, m, n, k int) (blocks, rowBytes, needA, needC, needW int, ok bool) {
	if m <= 0 || n <= 0 || k <= 0 || k%q8_0BlockElems != 0 {
		return 0, 0, 0, 0, 0, false
	}
	blocks = k / q8_0BlockElems
	rowBytes, ok = checked.MulInt(blocks, q8_0BlockBytes)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	needA, ok = checked.MulInt(m, k)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	needC, ok = checked.MulInt(m, n)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	needW, ok = checked.MulInt(n, rowBytes)
	if !ok || len(a) < needA || len(c) < needC || len(w) < needW {
		return 0, 0, 0, 0, 0, false
	}
	return blocks, rowBytes, needA, needC, needW, true
}

func q8_0FiniteScales(w []byte, blocks, n int) bool {
	for j := 0; j < n; j++ {
		row := w[j*blocks*q8_0BlockBytes:]
		for b := 0; b < blocks; b++ {
			d := half.F16ToF32(binary.LittleEndian.Uint16(row[b*q8_0BlockBytes:]))
			if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
				return false
			}
		}
	}
	return true
}

func float32ByteSlicesDisjoint(a []float32, b []byte) bool {
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap == bp {
		return false
	}
	asz, ok := checkedFloat32ByteOffset(len(a))
	if !ok {
		return false
	}
	aEnd := ap + asz
	bEnd := bp + uintptr(len(b))
	return aEnd <= bp || bEnd <= ap
}
