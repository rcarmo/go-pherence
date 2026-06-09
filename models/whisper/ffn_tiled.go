package whisper

import (
	"os"
	"strconv"
)

var ffnTileM = func() int {
	v := os.Getenv("WHISPER_FFN_TILE_M")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return n
}()

// forwardFFNTiled executes FC1, GELU, FC2, and residual on row blocks instead
// of materializing the full [seqLen, ffnDim] hidden buffer. It is opt-in via
// WHISPER_FFN_TILE_M and preserves the current FFN semantics unless an A100 FFN
// mode is explicitly enabled.
func forwardFFNTiled(layerIdx int, mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	if ffnTileM <= 0 || layer == nil || seqLen <= 0 {
		return nil, false
	}
	tileM := ffnTileM
	if tileM > seqLen {
		tileM = seqLen
	}
	out := make([]float32, seqLen*dModel)
	for start := 0; start < seqLen; start += tileM {
		end := start + tileM
		if end > seqLen {
			end = seqLen
		}
		m := end - start
		mlpTile := mlpIn[start*dModel : end*dModel]
		resTile := residual[start*dModel : end*dModel]
		var tileOut []float32
		var ok bool
		if tileOut, ok = forwardA100FFNTile(layerIdx, mlpTile, layer, resTile, m, dModel, ffnDim); !ok {
			hidden := linearForwardOpt(mlpTile, layer.FC1Weight, layer.FC1Bias, m, dModel, ffnDim)
			gelu(hidden)
			tileOut = linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, m, ffnDim, dModel)
			for i := range tileOut {
				tileOut[i] += resTile[i]
			}
		}
		copy(out[start*dModel:end*dModel], tileOut)
	}
	return out, true
}
