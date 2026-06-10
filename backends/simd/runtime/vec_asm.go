//go:build amd64 || arm64

package simd

func init() {
	HasVecAsm = RuntimeCapabilities().HasVec
}
