//go:build !riscv64

package rvv

import (
	"math"
	"unsafe"
)

// CopyBytesRVV is the non-RISC-V fallback for the RVV byte-copy primitive.
func CopyBytesRVV(dst, src *byte, n int) {
	if n <= 0 || dst == nil || src == nil {
		return
	}
	d := unsafeByteSlice(dst, n)
	s := unsafeByteSlice(src, n)
	copy(d, s)
}

// CopyTCMBytes copies src to dst. On RISC-V this is backed by RVV; elsewhere it
// is a normal Go copy so packages can still build and test on development hosts.
func CopyTCMBytes(dst, src []byte) {
	if len(dst) < len(src) {
		panic("rvv.CopyTCMBytes: len(dst) < len(src)")
	}
	copy(dst, src)
}

// F32ToF16RVV converts F32 to IEEE 754 binary16. Non-RISC-V fallback.
func F32ToF16RVV(dst []uint16, src []float32) {
	if len(dst) < len(src) {
		panic("rvv.F32ToF16RVV: len(dst) < len(src)")
	}
	for i, v := range src {
		dst[i] = f32ToF16Scalar(v)
	}
}

func DotF16(a, b []uint16) float32 {
	if len(a) == 0 {
		return 0
	}
	if len(b) < len(a) {
		panic("rvv.DotF16: len(b) < len(a)")
	}
	var sum float32
	for i := range a {
		sum += f16ToF32Scalar(a[i]) * f16ToF32Scalar(b[i])
	}
	return sum
}

func GemmF16(A, B []uint16, C []float32, M, N, K int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += f16ToF32Scalar(A[m*K+k]) * f16ToF32Scalar(B[n*K+k])
			}
			C[m*N+n] = sum
		}
	}
}

func PackBF16(B []uint16, N, K int) []uint16 { return packBF16TileFallback(B, N, K, 16) }
func PackBF16Into(B []uint16, N, K int, dst []uint16) []uint16 {
	return packBF16TileIntoFallback(B, N, K, 16, dst)
}
func PackBF16N32(B []uint16, N, K int) []uint16 { return packBF16TileFallback(B, N, K, 32) }
func PackBF16N32Into(B []uint16, N, K int, dst []uint16) []uint16 {
	return packBF16TileIntoFallback(B, N, K, 32, dst)
}

func GemmF16Outer(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16PackedFallback(A, Bp, C, M, N, K, 16)
}
func GemmF16Outer32(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16PackedFallback(A, Bp, C, M, N, K, 32)
}
func GemmF16Outer32Batch(nthreads int, specs ...GemmF16Outer32Spec) {
	for _, sp := range specs {
		GemmF16Outer32(sp.A, sp.Bp, sp.C, sp.M, sp.N, sp.K, 1)
	}
}
func GemmF16Threaded(A, B []uint16, C []float32, M, N, K, nthreads int) {
	GemmF16(A, B, C, M, N, K)
}

type GemmF16Outer32Spec struct {
	A, Bp   []uint16
	C       []float32
	M, N, K int
}

func dotI8(a, b *int8, n int64) int32 {
	if n <= 0 || a == nil || b == nil {
		return 0
	}
	as := unsafe.Slice(a, int(n))
	bs := unsafe.Slice(b, int(n))
	var sum int32
	for i := range as {
		sum += int32(as[i]) * int32(bs[i])
	}
	return sum
}

func GemmI8(A, B []int8, C []int32, M, N, K int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(A[m*K+k]) * int32(B[n*K+k])
			}
			C[m*N+n] = sum
		}
	}
}
func GemmI8Threaded(A, B []int8, C []int32, M, N, K, nthreads int) { GemmI8(A, B, C, M, N, K) }

func PackB(B []int8, N, K int) []int8 { return packI8TileFallback(B, N, K, 32) }
func GemmI8Outer(A, Bp []int8, C []int32, M, N, K, nthreads int) {
	gemmI8PackedFallback(A, Bp, C, M, N, K, 32)
}
func PackB4(B []int8, N, K int) []int8 { return packI8TileFallback(B, N, K, 32) }
func GemmI8OuterW4(A, Bp []int8, C []int32, M, N, K, nthreads int) {
	gemmI8PackedFallback(A, Bp, C, M, N, K, 32)
}
func fround(x float32) float32 {
	if x < 0 {
		return float32(int32(x - 0.5))
	}
	return float32(int32(x + 0.5))
}

func QuantizeDynamicU8(x []float32) (q []uint8, scale float32, zp uint8) {
	mn, mx := float32(0), float32(0)
	for _, v := range x {
		if v < mn {
			mn = v
		} else if v > mx {
			mx = v
		}
	}
	scale = (mx - mn) / 255
	if scale == 0 {
		scale = 1
	}
	zpf := fround(-mn / scale)
	if zpf < 0 {
		zpf = 0
	} else if zpf > 255 {
		zpf = 255
	}
	zp = uint8(zpf)
	q = make([]uint8, len(x))
	inv := float32(1) / scale
	for i, v := range x {
		r := fround(v*inv) + zpf
		if r < 0 {
			r = 0
		} else if r > 255 {
			r = 255
		}
		q[i] = uint8(r)
	}
	return
}

func QuantizeWeightsSym(W []float32, N, K int) (Wp []int8, wScale []float32, wColSum []int32) {
	Wq := make([]int8, N*K)
	wScale = make([]float32, N)
	wColSum = make([]int32, N)
	for n := 0; n < N; n++ {
		mx := float32(0)
		for k := 0; k < K; k++ {
			a := float32(math.Abs(float64(W[n*K+k])))
			if a > mx {
				mx = a
			}
		}
		s := mx / 127
		if s == 0 {
			s = 1
		}
		wScale[n] = s
		var cs int32
		for k := 0; k < K; k++ {
			q := int32(math.Round(float64(W[n*K+k] / s)))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			Wq[n*K+k] = int8(q)
			cs += q
		}
		wColSum[n] = cs
	}
	return PackB(Wq, N, K), wScale, wColSum
}

func gemmU8Outer(aq []uint8, Wp []int8, raw []int32, M, N, K, nthreads int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			nt, j := n/32, n%32
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(aq[m*K+k]) * int32(Wp[nt*K*32+k*32+j])
			}
			raw[m*N+n] = sum
		}
	}
}

func MatMulIntegerDequant(Af32 []float32, Wp []int8, wScale []float32, wColSum []int32, C []float32, M, N, K, nthreads int) {
	aq, aScale, aZp := QuantizeDynamicU8(Af32)
	raw := make([]int32, M*N)
	gemmU8Outer(aq, Wp, raw, M, N, K, nthreads)
	zpf := int32(aZp)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			corr := raw[m*N+n] - zpf*wColSum[n]
			C[m*N+n] = aScale * wScale[n] * float32(corr)
		}
	}
}

func PackBW4(W []int8, N, K int) []int8 {
	out := make([]int8, N/32*K*16)
	for nt := 0; nt < N/32; nt++ {
		base := nt * K * 16
		for k := 0; k < K; k++ {
			for j := 0; j < 16; j++ {
				lo := W[(nt*32+j)*K+k] & 0xF
				hi := W[(nt*32+16+j)*K+k] & 0xF
				out[base+k*16+j] = lo | (hi << 4)
			}
		}
	}
	return out
}

func GemmU8W4(aq []uint8, B4 []int8, C []int32, M, N, K, nthreads int) {
	gemmU8W4Fallback(aq, B4, C, M, N, K)
}

func gemmU8W4Fallback(aq []uint8, B4 []int8, C []int32, M, N, K int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			nt, j := n/32, n%32
			var sum int32
			for k := 0; k < K; k++ {
				packed := B4[nt*K*16+k*16+j%16]
				var w int8
				if j < 16 {
					w = packed & 0x0f
				} else {
					w = (packed >> 4) & 0x0f
				}
				if w >= 8 {
					w -= 16
				}
				sum += int32(aq[m*K+k]) * int32(w)
			}
			C[m*N+n] = sum
		}
	}
}

func QuantizeQ8Block32RVV(src *float32, dst *byte, divisor *float32) {
	if src == nil || dst == nil || divisor == nil {
		return
	}
	s := unsafeFloat32Slice(src, 32)
	d := unsafeByteSlice(dst, 32)
	div := *divisor
	if div == 0 {
		div = 1
	}
	for i, v := range s {
		q := int(math.Round(float64(v / div)))
		if q < -128 {
			q = -128
		}
		if q > 127 {
			q = 127
		}
		d[i] = byte(int8(q))
	}
}

func unsafeByteSlice(p *byte, n int) []byte {
	return unsafe.Slice(p, n)
}

func unsafeFloat32Slice(p *float32, n int) []float32 {
	return unsafe.Slice(p, n)
}

func f32ToF16Scalar(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	mant := bits & 0x7fffff
	if exp == 255 {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := uint16(mant >> shift)
		if (mant>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 {
		half++
	}
	return half
}

func f16ToF32Scalar(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		fallthrough
	case exp < 31:
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	default:
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}
}

func packBF16Tile(B []uint16, N, K, tileN int) []uint16 {
	return packBF16TileFallback(B, N, K, tileN)
}

func packBF16TileInto(B []uint16, N, K, tileN int, dst []uint16) []uint16 {
	return packBF16TileIntoFallback(B, N, K, tileN, dst)
}

func packBF16TileFallback(B []uint16, N, K, tileN int) []uint16 {
	return packBF16TileIntoFallback(B, N, K, tileN, make([]uint16, N*K))
}
func packBF16TileIntoFallback(B []uint16, N, K, tileN int, dst []uint16) []uint16 {
	Bp := dst[:N*K]
	for nt := 0; nt < N/tileN; nt++ {
		base := nt * K * tileN
		for k := 0; k < K; k++ {
			for j := 0; j < tileN; j++ {
				Bp[base+k*tileN+j] = B[(nt*tileN+j)*K+k]
			}
		}
	}
	return Bp
}
func packI8TileFallback(B []int8, N, K, tileN int) []int8 {
	Bp := make([]int8, N*K)
	for nt := 0; nt < N/tileN; nt++ {
		base := nt * K * tileN
		for k := 0; k < K; k++ {
			for j := 0; j < tileN; j++ {
				Bp[base+k*tileN+j] = B[(nt*tileN+j)*K+k]
			}
		}
	}
	return Bp
}
func gemmI8PackedFallback(A, Bp []int8, C []int32, M, N, K, tileN int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			nt, j := n/tileN, n%tileN
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(A[m*K+k]) * int32(Bp[nt*K*tileN+k*tileN+j])
			}
			C[m*N+n] = sum
		}
	}
}
func gemmF16PackedFallback(A, Bp []uint16, C []float32, M, N, K, tileN int) {
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			nt, j := n/tileN, n%tileN
			var sum float32
			for k := 0; k < K; k++ {
				sum += f16ToF32Scalar(A[m*K+k]) * f16ToF32Scalar(Bp[nt*K*tileN+k*tileN+j])
			}
			C[m*N+n] = sum
		}
	}
}
