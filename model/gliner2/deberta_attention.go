package gliner2

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
)

// RelativePositionBucket matches DeBERTa-v2's signed logarithmic buckets.
func RelativePositionBucket(relative, buckets, maxPosition int) (int, error) {
	if buckets < 2 || buckets%2 != 0 || maxPosition <= buckets/2+1 {
		return 0, fmt.Errorf("invalid relative bucket configuration")
	}
	mid := buckets / 2
	a := absInt(relative)
	if a <= mid {
		return relative, nil
	}
	mapped := int(math.Ceil(math.Log(float64(a)/float64(mid))/math.Log(float64(maxPosition-1)/float64(mid))*float64(mid-1))) + mid
	if relative < 0 {
		mapped = -mapped
	}
	return mapped, nil
}

// DisentangledAttention implements the published encoder's share_att_key=true
// c2p+p2c attention, operating on already projected Q/K/V and relative Q/K.
// All projections are token-major, with contiguous heads within each row.
func DisentangledAttention(query, key, value, relativeQuery, relativeKey [][]float32, mask []bool, heads, buckets, maxPosition int) ([][]float32, error) {
	n := len(query)
	if n == 0 || len(key) != n || len(value) != n || len(mask) != n || heads <= 0 {
		return nil, fmt.Errorf("invalid attention rows")
	}
	width := len(query[0])
	if width == 0 || width%heads != 0 {
		return nil, fmt.Errorf("invalid attention width")
	}
	if _, err := RelativePositionBucket(0, buckets, maxPosition); err != nil {
		return nil, err
	}
	if len(relativeQuery) != 2*buckets || len(relativeKey) != 2*buckets {
		return nil, fmt.Errorf("relative embedding row mismatch")
	}
	for _, rows := range [][][]float32{query, key, value, relativeQuery, relativeKey} {
		for _, row := range rows {
			if len(row) != width {
				return nil, fmt.Errorf("attention row width mismatch")
			}
		}
	}
	dim := width / heads
	scale := float32(1 / math.Sqrt(float64(dim*3)))
	out := make([][]float32, n)
	for i := range query {
		out[i] = make([]float32, width)
		for h := 0; h < heads; h++ {
			lo, hi := h*dim, (h+1)*dim
			scores := make([]float32, n)
			for j := range key {
				// The encoder mask is the outer product of query/key validity. Fully
				// masked query rows retain PyTorch's uniform finite-min softmax.
				if !mask[i] || !mask[j] {
					scores[j] = -math.MaxFloat32
					continue
				}
				rel, _ := RelativePositionBucket(i-j, buckets, maxPosition)
				idx := max(0, min(rel+buckets, 2*buckets-1))
				// p2c gathers -rpos then transposes: at [i,j] it uses rel(i,j).
				scores[j] = (simd.Sdot(query[i][lo:hi], key[j][lo:hi]) + simd.Sdot(query[i][lo:hi], relativeKey[idx][lo:hi]) + simd.Sdot(key[j][lo:hi], relativeQuery[idx][lo:hi])) * scale
			}
			simd.SoftmaxInPlace(scores)
			for j, p := range scores {
				for d := lo; d < hi; d++ {
					out[i][d] += p * value[j][d]
				}
			}
		}
	}
	return out, nil
}
