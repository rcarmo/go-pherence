package quant

import (
	"github.com/rcarmo/go-pherence/backends/mlx"
)

type MLXQuantWeight = mlx.QuantWeight

func DequantMLX(qw *MLXQuantWeight) []float32             { return mlx.Dequant(qw) }
func DequantMLXTo(out []float32, qw *MLXQuantWeight) bool { return mlx.DequantTo(out, qw) }
func GemvMLQ(out, x []float32, qw *MLXQuantWeight)        { mlx.Gemv(out, x, qw) }
func GemvMLQTo(out, x []float32, qw *MLXQuantWeight) bool { return mlx.GemvTo(out, x, qw) }
func GemmMLQ(out, x []float32, batch int, qw *MLXQuantWeight) bool {
	return mlx.Gemm(out, x, batch, qw)
}
func ValidateMLXQuantWeight(qw *MLXQuantWeight) error { return mlx.ValidateQuantWeight(qw) }
func LoadMLXWeight(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, prefix string, outDim, inDim, groupSize, bits int) (*MLXQuantWeight, error) {
	return mlx.LoadWeight(f, prefix, outDim, inDim, groupSize, bits)
}
