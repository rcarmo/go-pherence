package main
import (
	"fmt"
	"math"
	"os"
	"github.com/rcarmo/go-pherence/loader/gguf"
)
func main() {
	g, _ := gguf.Open(os.Args[1])
	
	// Check output_norm
	normT, _ := g.TensorByName("output_norm.weight")
	normF32, _ := g.DequantF32(normT)
	fmt.Printf("output_norm[0:8]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
		normF32[0], normF32[1], normF32[2], normF32[3], normF32[4], normF32[5], normF32[6], normF32[7])
	fmt.Printf("output_norm rms: %.4f\n", rms(normF32))
	
	// Check embedding for token 2501
	tokT, _ := g.TensorByName("token_embd.weight")
	tokF32, _ := g.DequantF32(tokT)
	nEmbd := int(tokT.Shape[0])
	emb := tokF32[2501*nEmbd : 2502*nEmbd]
	fmt.Printf("embd[2501] rms: %.6f\n", rms(emb))
	
	// Check output.weight row 0
	outT, _ := g.TensorByName("output.weight")
	outF32, _ := g.DequantF32(outT)
	fmt.Printf("output[0] rms: %.6f\n", rms(outF32[:nEmbd]))
	fmt.Printf("output[263] rms: %.6f\n", rms(outF32[263*nEmbd:(263+1)*nEmbd]))
	
	// Compute logit[0] and logit[263] manually with 0 layers
	// norm(emb[2501])
	xn := make([]float32, nEmbd)
	r := float32(1.0 / math.Sqrt(float64(dotSelf(emb)/float32(nEmbd) + 1e-6)))
	for i := range xn { xn[i] = emb[i] * r * normF32[i] }
	fmt.Printf("normed_emb rms: %.6f\n", rms(xn))
	
	// logit[0] = dot(output[0], xn)
	var l0, l263 float32
	for k := 0; k < nEmbd; k++ { l0 += outF32[k] * xn[k] }
	for k := 0; k < nEmbd; k++ { l263 += outF32[263*nEmbd+k] * xn[k] }
	fmt.Printf("logit[0]=%.4f logit[263]=%.4f\n", l0, l263)
}
func rms(x []float32) float32 {
	var s float64
	for _, v := range x { s += float64(v) * float64(v) }
	return float32(math.Sqrt(s / float64(len(x))))
}
func dotSelf(x []float32) float32 {
	var s float32
	for _, v := range x { s += v * v }
	return s
}
