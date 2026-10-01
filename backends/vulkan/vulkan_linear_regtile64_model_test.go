package vulkan

import (
	"math"
	"math/rand"
	"testing"
)

// Simulates shuffled group/lane ownership, staged shared loads, all sixteen
// accumulators and the K32 barriers independently of the executing shader.
func TestVulkanOfflineLinearRegTile64Ownership(t *testing.T) {
	for _, d := range [][3]int{{1, 1, 1}, {2, 3, 5}, {63, 31, 65}, {65, 33, 63}, {64, 64, 64}, {67, 65, 69}} {
		m, k, n := d[0], d[1], d[2]
		x, w, b := nativeData(m*k, 11, .5), nativeData(n*k, 12, .5), nativeData(n, 13, .1)
		ref := nativeLinearRef(x, w, b, m, k, n)
		for seed := int64(0); seed < 4; seed++ {
			rng := rand.New(rand.NewSource(seed))
			lanes := rng.Perm(256)
			gx, gy := (n+63)/64, (m+63)/64
			out := make([]float32, m*n)
			writers := make([]int, m*n)
			for _, group := range rng.Perm(gx * gy) {
				rowBase, colBase := (group/gx)*64, (group%gx)*64
				var sum [256][16]float32
				for base := 0; base < k; base += 32 {
					var xt, wt [2048]float32
					var owners [2048]int
					for _, lane := range lanes {
						for i := lane; i < 2048; i += 256 {
							r, kk := i/32, base+i%32
							owners[i]++
							if kk < k {
								if rowBase+r < m {
									xt[i] = x[(rowBase+r)*k+kk]
								}
								if colBase+r < n {
									wt[i] = w[(colBase+r)*k+kk]
								}
							}
						}
					}
					for _, owner := range owners {
						if owner != 1 {
							t.Fatal("shared ownership", owner)
						}
					}
					for _, lane := range lanes {
						cx, ry := lane%16, lane/16
						for kk := 0; kk < 32; kk++ {
							for r := 0; r < 4; r++ {
								a := xt[(ry+r*16)*32+kk]
								for c := 0; c < 4; c++ {
									sum[lane][r*4+c] += a * wt[(cx+c*16)*32+kk]
								}
							}
						}
					}
				}
				for _, lane := range lanes {
					cx, ry := lane%16, lane/16
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							rr, cc := rowBase+ry+r*16, colBase+cx+c*16
							if rr < m && cc < n {
								out[rr*n+cc] = sum[lane][r*4+c] + b[cc]
								writers[rr*n+cc]++
							}
						}
					}
				}
			}
			old, _ := linearTileModel(x, w, b, m, k, n, seed)
			for i, v := range out {
				if writers[i] != 1 || math.Float32bits(v) != math.Float32bits(old[i]) || math.Abs(float64(v)-ref[i]) > 2e-5+2e-5*math.Abs(ref[i]) {
					t.Fatal("output ownership/arithmetic", d, seed, i, writers[i], v, old[i], ref[i])
				}
			}
		}
	}
}
