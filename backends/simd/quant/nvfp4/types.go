package nvfp4

// NVFP4Weight holds the TensorRT Model Optimizer / NVFP4 safetensors layout
// seen in public Qwen3 and Gemma4 checkpoints.
//
// Observed tensor set for quantized linear weights:
//
//	<prefix>.weight         U8       [outDim, inDim/2] packed FP4, two weights/byte
//	<prefix>.weight_scale   F8_E4M3  [outDim, inDim/groupSize] per-block scale bytes
//	<prefix>.weight_scale_2 F32      [] global/secondary scale
//	<prefix>.input_scale    F32      [] activation scale, optional for some tensors
type NVFP4Weight struct {
	Weight        []byte  // [outDim, inDim/2] U8 packed FP4 nibbles
	WeightScale   []byte  // [outDim, groups] raw F8_E4M3 bytes
	WeightScale2  float32 // scalar secondary/global scale
	InputScale    float32 // optional scalar activation scale
	HasInputScale bool
	OutDim        int
	InDim         int
	Groups        int
	GroupSize     int // observed ModelOpt NVFP4 group size is 16
}
