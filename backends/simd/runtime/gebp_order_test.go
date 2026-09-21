package simd

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// Reports a deterministic fingerprint to compare same-host kernel revisions.
// It is intentionally not hardcoded: fallback and architecture accumulation
// orders need not match each other. Includes nonzero C, alpha, tails and strides.
func TestGEBPOrderFingerprint(t *testing.T) {
	hash := sha256.New()
	var raw [4]byte
	for _, k := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1023, 1024, 3072} {
		for _, m := range []int{1, 6, 7, 13} {
			for _, n := range []int{16, 19, 32} {
				lda, ldb, ldc := k+3, k+5, n+7
				a, w, c := make([]float32, (m-1)*lda+k), make([]float32, (n-1)*ldb+k), make([]float32, (m-1)*ldc+n)
				for i := range a {
					a[i] = float32(math.Sin(float64(i) * .31))
				}
				for i := range w {
					w[i] = float32(math.Cos(float64(i) * .17))
				}
				for i := range c {
					c[i] = float32(i%19-9) * .03
				}
				scratch := make([]float32, k*16)
				if !SgemmNTPackedTo(c, a, w, scratch, m, n, k, -.75, lda, ldb, ldc) {
					t.Fatal("rejected valid input")
				}
				for _, v := range c {
					binary.LittleEndian.PutUint32(raw[:], math.Float32bits(v))
					hash.Write(raw[:])
				}
			}
		}
	}
	fmt.Printf("GEBP fingerprint: %x\n", hash.Sum(nil))
}
