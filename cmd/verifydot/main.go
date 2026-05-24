package main

import (
	"fmt"
	"math"
	"os"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	g, _ := gguf.Open(os.Args[1])
	tokT, _ := g.TensorByName("token_embd.weight")
	tokF32, _ := g.DequantF32(tokT)
	nEmbd := int(tokT.Shape[0])
	
	// Token 374 ("is") embedding
	tok := 374
	fmt.Printf("token_embd[%d][0:8]:", tok)
	for i := 0; i < 8; i++ {
		fmt.Printf(" %.6f", tokF32[tok*nEmbd+i])
	}
	fmt.Println()
	
	// Also check: what does RMSnorm of this embedding give?
	normT, _ := g.TensorByName("output_norm.weight")
	normF32, _ := g.DequantF32(normT)
	
	x := tokF32[tok*nEmbd : (tok+1)*nEmbd]
	var ss float32
	for i := 0; i < nEmbd; i++ { ss += x[i] * x[i] }
	ss = 1.0 / float32(len(x))  // wrong - need sqrt
	
	// Compute logit for token 374 (self-dot through norm)
	// logit[374] = dot(tokEmbd[374], norm(tokEmbd[374]))
	// First compute norm(x):
	var rms float32
	for i := 0; i < nEmbd; i++ { rms += x[i] * x[i] }
	
	rms = float32(1.0 / math.Sqrt(float64(rms/float32(nEmbd) + 1e-5)))
	xn := make([]float32, nEmbd)
	for i := 0; i < nEmbd; i++ { xn[i] = x[i] * rms * normF32[i] }
	
	// Self-dot
	var selfDot float32
	for i := 0; i < nEmbd; i++ { selfDot += tokF32[tok*nEmbd+i] * xn[i] }
	fmt.Printf("logit[374] (self-dot after norm) = %.4f\n", selfDot)
	
	// Also compute logit[264] ("a") for comparison
	var dot264 float32
	for i := 0; i < nEmbd; i++ { dot264 += tokF32[264*nEmbd+i] * xn[i] }
	fmt.Printf("logit[264] = %.4f\n", dot264)
	
	// And logit[635]
	var dot635 float32
	for i := 0; i < nEmbd; i++ { dot635 += tokF32[635*nEmbd+i] * xn[i] }
	fmt.Printf("logit[635] = %.4f\n", dot635)
}
