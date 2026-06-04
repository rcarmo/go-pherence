package ime2

import (
	"math"
	"unsafe"
)

// rvvBroadcastPack copies K int8 values into 4× broadcast tile format.
//
//go:noescape
func rvvBroadcastPack(src *byte, K int, dst *byte)

// rvvMulVecVec computes out[i] = a[i] * b[i] using RVV.
//
//go:noescape
func rvvMulVecVec(a *float32, b *float32, out *float32, n int)

// RMSNormRVV computes out[i] = x[i] * invRMS * weight[i] with zero allocation.
// Modifies x in-place (scales by invRMS) then uses RVV for x*weight.
// After return, x is scaled (caller should treat as consumed).
func RMSNormRVV(x, weight, out []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		x[i] *= invRMS
	}
	rvvMulVecVec(&x[0], &weight[0], &out[0], n)
}

// RMSNormFast is RMSNorm without modifying x (uses out as temp).
func RMSNormFast(x, weight, out []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	// Scale into out first, then multiply by weight in-place
	for i := 0; i < n; i++ {
		out[i] = x[i] * invRMS
	}
	rvvMulVecVec(&out[0], &weight[0], &out[0], n)
}

// BroadcastPackRVV packs K int8 values into 4× tile format using RVV.
func BroadcastPackRVV(src []int8, K int, dst []int8) {
	rvvBroadcastPack((*byte)(unsafe.Pointer(&src[0])), K, (*byte)(unsafe.Pointer(&dst[0])))
}

// fusedMatVec performs quantize+vmadot+dequant in one assembly function.
// out[i] = sum_k(wPacked[i,k] * quantize(act[k])) * wScale * actInvScale
// M must be multiple of 4. K must be multiple of 8.
//go:noescape

// FusedMatVec is the exported wrapper.

// fusedPackVmadot fuses broadcast-pack + vmadot K-loop.
//
//go:noescape
func fusedPackVmadot(wPacked *byte, actI8 *byte, M int, K int, out *int32)

// FusedPackVmadot is the exported wrapper.
// Performs C[M] = wPacked[M×K] · actI8[K] (int8 matmul, int32 output at stride 4).
func FusedPackVmadot(M, K int, wPacked []int8, actI8 []int8, out []int32) {
	rawOut := make([]int32, M*4)
	fusedPackVmadot((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actI8[0])), M, K, &rawOut[0])
	for i := 0; i < M; i++ {
		out[i] = rawOut[i*4]
	}
}

// rvvFindMaxAbs returns max(|x[i]|) using RVV vectorized reduction.
//
//go:noescape
func rvvFindMaxAbs(x *float32, n int) float32

// FindMaxAbsRVV is the exported wrapper.
func FindMaxAbsRVV(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	return rvvFindMaxAbs(&x[0], len(x))
}

// rvvQuantizeF32ToI8 quantizes float32 to int8 using RVV.
//
//go:noescape
func rvvQuantizeF32ToI8(src *float32, scaleBits uint32, dst *byte, n int)

// QuantizeF32ToI8RVV quantizes n float32s to int8 with the given scale.
func QuantizeF32ToI8RVV(src []float32, scale float32, dst []int8) {
	bits := math.Float32bits(scale)
	rvvQuantizeF32ToI8(&src[0], bits, (*byte)(unsafe.Pointer(&dst[0])), len(src))
}

// vmadotQ4KPackedLoop reads packed Q4K bytes and unpacks inline with RVV.
// 2× less weight bandwidth. actReord must be in even/odd interleaved format.
//
//go:noescape
func vmadotQ4KPackedLoop(wQS *byte, actReord *byte, acc *int32, Kgroups int)

// ReorderActEvenOdd reorders activation for packed Q4K vmadot.
// Output: [act[0],act[2],...,act[30], act[1],act[3],...,act[31]] per 32-element group.
func ReorderActEvenOdd(act []int8, K int) []int8 {
	out := make([]int8, K)
	for g := 0; g < K; g += 32 {
		for i := 0; i < 16; i++ {
			out[g+i] = act[g+2*i]      // even elements first
			out[g+16+i] = act[g+2*i+1] // odd elements second
		}
	}
	return out
}

// PackQ4KForI2K stores raw Q4K qs bytes in tile format (16 bytes per 32-element group per row).
// For M rows × K elements: output is M/4 row-groups × K/32 groups × 64 bytes
// (4 rows × 16 packed bytes per row per group).
func PackQ4KForI2K(rawQS []byte, M, K int) []byte {
	bytesPerGroup := 16 // per row (32 nibbles = 16 packed bytes)
	groupsPerRow := K / 32
	out := make([]byte, (M/4)*groupsPerRow*64) // 4 rows × 16 bytes = 64 per tile
	for rg := 0; rg < M; rg += 4 {
		for gi := 0; gi < groupsPerRow; gi++ {
			tileOff := ((rg/4)*groupsPerRow + gi) * 64
			for r := 0; r < 4; r++ {
				srcOff := (rg+r)*(K/2) + gi*bytesPerGroup
				copy(out[tileOff+r*16:tileOff+(r+1)*16], rawQS[srcOff:srcOff+16])
			}
		}
	}
	return out
}

// vmadotKLoopAI forces vl=32 on VLEN=1024 cores (128-byte tiles).
//
//go:noescape
func vmadotKLoopAI(A *byte, B *byte, C *int32, K int)

// VmadotKLoopAI is the exported wrapper.
func VmadotKLoopAI(A *byte, B *byte, C *int32, K int) { vmadotKLoopAI(A, B, C, K) }

// PackTiles1024 packs int8 matrix into 128-byte tiles for native VLEN=1024.
// Probed semantics: vmadot forms an 8×8 accumulator, where each source group
// is 16 bytes. Each tile is therefore 8 rows × 16 columns.
func PackTiles1024(src []int8, rows, K int) []int8 {
	if rows%8 != 0 || K%16 != 0 {
		panic("ime2: PackTiles1024 requires rows%8==0, K%16==0")
	}
	dst := make([]int8, rows*K)
	for rg := 0; rg < rows; rg += 8 {
		for ki := 0; ki < K; ki += 16 {
			tileIdx := (rg/8)*(K/16) + ki/16
			tileBase := tileIdx * 128
			for r := 0; r < 8; r++ {
				copy(dst[tileBase+r*16:tileBase+(r+1)*16],
					src[(rg+r)*K+ki:(rg+r)*K+ki+16])
			}
		}
	}
	return dst
}

// BroadcastPack1024 packs K int8 values into native VLEN=1024 broadcast tile
// format. Each tile = 8 copies of 16 consecutive bytes = 128 bytes.
func BroadcastPack1024(src []int8, K int) []int8 {
	if K%16 != 0 {
		panic("ime2: BroadcastPack1024 requires K%16==0")
	}
	dst := make([]int8, 8*K) // K/16 tiles × 128 bytes = 8*K
	for ki := 0; ki < K; ki += 16 {
		tileBase := (ki / 16) * 128
		for r := 0; r < 8; r++ {
			copy(dst[tileBase+r*16:tileBase+(r+1)*16], src[ki:ki+16])
		}
	}
	return dst
}

// VmadotKLoop1024 uses native vl=128 (full VLEN=1024). For probing only.
func VmadotKLoop1024(A *byte, B *byte, C *int32, K int) {
	vmadotKLoop1024native(A, B, C, K)
}

//go:noescape
func vmadotKLoop1024native(A *byte, B *byte, C *int32, K int)

// GemmINT8_AI performs matmul on AI cores (VLEN=1024).
// M output rows, K input dimension. wPacked in 128-byte tile format.
// actPacked in 128-byte broadcast tile format.
// Output: out[M] float32 (scaled by wScale * actScale).
func GemmINT8_AI(M, K int, wPacked []int8, actPacked []int8, wScale, actScale float32, out []float32) {
	tilesPerRow := K / 16
	combined := wScale * actScale
	for i := 0; i < M; i += 8 {
		var acc [64]int32
		VmadotKLoop1024(
			(*byte)(unsafe.Pointer(&wPacked[(i/8)*tilesPerRow*128])),
			(*byte)(unsafe.Pointer(&actPacked[0])),
			&acc[0], K,
		)
		// With broadcast activation, all accumulator columns are identical.
		// Output row r is acc[r][0] = acc[r*8].
		for r := 0; r < 8 && i+r < M; r++ {
			out[i+r] = float32(acc[r*8]) * combined
		}
	}
}

// rvvDotF32 computes sum(a[i]*b[i]) using RVV e32,m4.
//
//go:noescape
func rvvDotF32(a *float32, b *float32, n int) float32

// rvvScaleAccF32 computes acc[i] += scale*src[i] using RVV e32,m4.
//
//go:noescape
func rvvScaleAccF32(acc *float32, src *float32, scale float32, n int)

// DotF32RVV returns the dot product of a and b using SIMD.
func DotF32RVV(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return rvvDotF32(&a[0], &b[0], len(a))
}

// ScaleAccF32RVV computes acc[i] += scale*src[i] in-place using SIMD.
func ScaleAccF32RVV(acc, src []float32, scale float32) {
	if len(acc) == 0 || scale == 0 {
		return
	}
	rvvScaleAccF32(&acc[0], &src[0], scale, len(acc))
}
