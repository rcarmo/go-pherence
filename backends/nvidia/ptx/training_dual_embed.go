package ptx

import _ "embed"

// TrainingDualPTX is compiled from the adjacent readable CUDA source using
// nvcc12.8 (PTX8.7,sm80, no fast math/FMA contraction). It requires a compatible
// driver; do not silently substitute the older PTX7 module on load failure.
//
//go:embed training_dual.ptx
var TrainingDualPTX string
