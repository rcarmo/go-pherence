package community1

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// LSTMConfig describes an unprojected, inference-only PyTorch LSTM. Dropout is
// disabled in evaluation. Geometry must come from the checkpoint, not PyanNet
// constructor defaults. Inputs/outputs are one window, batch size one.
type LSTMConfig struct {
	InputSize, HiddenSize, NumLayers int
	Bidirectional                    bool
}

// LSTMWeights stores row-major PyTorch IFGO parameters for one direction.
// WeightIH is [4*hidden,input], WeightHH [4*hidden,hidden]; both biases [4*hidden].
// Later layer inputs concatenate forward then reverse outputs at the SAME frame.
type LSTMWeights struct{ WeightIH, WeightHH, BiasIH, BiasHH []float32 }
type LSTMLayer struct{ Forward, Reverse LSTMWeights }

// LSTM owns immutable finite weights. Independent calls may share the model;
// all recurrent state/scratch is call-local. No source arrays are retained.
type LSTM struct {
	cfg    LSTMConfig
	layers []LSTMLayer
}

// LSTMMode makes scalar comparison versus existing checked Plan9 SIMD dispatch
// explicit. SIMD mode reuses GemvRows; this is not a new specialised kernel or
// a measured speedup. Scalar order is a float32 reference, not bit-exact PyTorch.
type LSTMMode uint8

const (
	LSTMScalar LSTMMode = iota
	LSTMSIMD
)

// LSTMResult owns sequence Output [frames,directions*hidden] plus terminal
// Hidden/Cell [layers*directions,hidden], direction order forward then reverse.
// Reverse terminal state is after processing frame zero, not the last row of
// the concatenated sequence. No state is retained in the model between calls.
type LSTMResult struct{ Output, Hidden, Cell []float32 }

// LSTMObserver sees each completed layer [frames,directions*hidden]. Views are
// read-only and valid only during the synchronous callback. Copy to retain them.
type LSTMObserver func(layer, frames, features int, values []float32)

func lstmDirections(c LSTMConfig) int {
	if c.Bidirectional {
		return 2
	}
	return 1
}
func checkLSTMConfig(c LSTMConfig) error {
	if c.InputSize < 1 || c.InputSize > 512 || c.HiddenSize < 1 || c.HiddenSize > 256 || c.NumLayers < 1 || c.NumLayers > 4 {
		return fmt.Errorf("LSTM geometry exceeds input1..512/hidden1..256/layers1..4 bounds")
	}
	return nil
}
func finiteLSTM(ctx context.Context, values []float32) error {
	for i, v := range values {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite LSTM values")
		}
	}
	return ctx.Err()
}
func emptyLSTMWeights(w LSTMWeights) bool {
	return len(w.WeightIH)+len(w.WeightHH)+len(w.BiasIH)+len(w.BiasHH) == 0
}

// NewLSTM validates every shape/value before copying weights. Caller buffers
// must be immutable during construction. No tensor loader, GPU or CGo is used.
func NewLSTM(ctx context.Context, cfg LSTMConfig, layers []LSTMLayer) (*LSTM, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := checkLSTMConfig(cfg); err != nil {
		return nil, err
	}
	if len(layers) != cfg.NumLayers {
		return nil, fmt.Errorf("LSTM layer count mismatch")
	}
	directions := lstmDirections(cfg)
	for index, layer := range layers {
		width := cfg.InputSize
		if index > 0 {
			width = directions * cfg.HiddenSize
		}
		weights := []LSTMWeights{layer.Forward}
		if directions == 2 {
			weights = append(weights, layer.Reverse)
		} else if !emptyLSTMWeights(layer.Reverse) {
			return nil, fmt.Errorf("reverse weights on unidirectional LSTM")
		}
		for direction, w := range weights {
			if len(w.WeightIH) != 4*cfg.HiddenSize*width || len(w.WeightHH) != 4*cfg.HiddenSize*cfg.HiddenSize || len(w.BiasIH) != 4*cfg.HiddenSize || len(w.BiasHH) != 4*cfg.HiddenSize {
				return nil, fmt.Errorf("invalid LSTM layer%d direction%d IFGO shapes", index, direction)
			}
			for _, values := range [][]float32{w.WeightIH, w.WeightHH, w.BiasIH, w.BiasHH} {
				if err := finiteLSTM(ctx, values); err != nil {
					return nil, err
				}
			}
		}
	}
	model := &LSTM{cfg: cfg, layers: make([]LSTMLayer, len(layers))}
	clone := func(w LSTMWeights) (LSTMWeights, error) {
		var owned LSTMWeights
		for _, pair := range []struct {
			src []float32
			dst *[]float32
		}{{w.WeightIH, &owned.WeightIH}, {w.WeightHH, &owned.WeightHH}, {w.BiasIH, &owned.BiasIH}, {w.BiasHH, &owned.BiasHH}} {
			*pair.dst = make([]float32, len(pair.src))
			for start := 0; start < len(pair.src); start += 4096 {
				if err := ctx.Err(); err != nil {
					return LSTMWeights{}, err
				}
				end := min(start+4096, len(pair.src))
				copy((*pair.dst)[start:end], pair.src[start:end])
			}
		}
		return owned, nil
	}
	for index, layer := range layers {
		var err error
		model.layers[index].Forward, err = clone(layer.Forward)
		if err != nil {
			return nil, err
		}
		if directions == 2 {
			model.layers[index].Reverse, err = clone(layer.Reverse)
			if err != nil {
				return nil, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return model, nil
}

// Forward evaluates a complete window. It is NOT causal streaming when
// bidirectional: reverse recurrence requires the full window. Initial hidden
// and cell must both be nil (zeros), or both [layers*directions,hidden]. Frames
// must be 1..4096. Input/initial buffers are read-only, immutable during the call.
// Per-call memory is window bounded; no per-frame allocation or worker exists.
// Cancellation is checked before/after projections and each recurrence step;
// a running GEMV or callback is synchronous and cannot be interrupted. Errors
// return no partial result; observers may already have seen completed layers.
func (m *LSTM) Forward(ctx context.Context, input []float32, frames int, hidden, cell []float32, mode LSTMMode) (*LSTMResult, error) {
	return m.ForwardObserved(ctx, input, frames, hidden, cell, mode, nil)
}

func (m *LSTM) ForwardObserved(ctx context.Context, input []float32, frames int, hidden, cell []float32, mode LSTMMode, observe LSTMObserver) (*LSTMResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, fmt.Errorf("nil LSTM")
	}
	if err := checkLSTMConfig(m.cfg); err != nil {
		return nil, err
	}
	c := m.cfg
	dirs := lstmDirections(c)
	h := c.HiddenSize
	stateSize := c.NumLayers * dirs * h
	if len(m.layers) != c.NumLayers || frames < 1 || frames > 4096 || len(input) != frames*c.InputSize || (mode != LSTMScalar && mode != LSTMSIMD) {
		return nil, fmt.Errorf("invalid LSTM input/mode")
	}
	if !(hidden == nil && cell == nil) && (len(hidden) != stateSize || len(cell) != stateSize) {
		return nil, fmt.Errorf("invalid initial LSTM state")
	}
	for _, values := range [][]float32{input, hidden, cell} {
		if err := finiteLSTM(ctx, values); err != nil {
			return nil, err
		}
	}
	result := &LSTMResult{Hidden: make([]float32, stateSize), Cell: make([]float32, stateSize)}
	copy(result.Hidden, hidden)
	copy(result.Cell, cell)
	// Two alternating sequence buffers suffice for any supported layer count.
	first := make([]float32, frames*dirs*h)
	var second []float32
	if c.NumLayers > 1 {
		second = make([]float32, len(first))
	}
	gateIH, gateHH := make([]float32, 4*h), make([]float32, 4*h)
	sequence := input
	width := c.InputSize
	for layerIndex, layer := range m.layers {
		out := first
		if layerIndex%2 != 0 {
			out = second
		}
		for direction := 0; direction < dirs; direction++ {
			w := layer.Forward
			if direction == 1 {
				w = layer.Reverse
			}
			base := (layerIndex*dirs + direction) * h
			hs, cs := result.Hidden[base:base+h], result.Cell[base:base+h]
			for step := 0; step < frames; step++ {
				frame := step
				if direction == 1 {
					frame = frames - 1 - step
				}
				if err := lstmProjection(ctx, gateIH, sequence[frame*width:(frame+1)*width], w.WeightIH, 4*h, width, mode); err != nil {
					return nil, err
				}
				if err := lstmProjection(ctx, gateHH, hs, w.WeightHH, 4*h, h, mode); err != nil {
					return nil, err
				}
				for i := range gateIH {
					// Preserve distinct input/hidden bias additions (PyTorch IFGO).
					gateIH[i] = (gateIH[i] + w.BiasIH[i]) + (gateHH[i] + w.BiasHH[i])
					if math.IsNaN(float64(gateIH[i])) || math.IsInf(float64(gateIH[i]), 0) {
						return nil, fmt.Errorf("non-finite LSTM affine output")
					}
				}
				for unit := 0; unit < h; unit++ {
					i := lstmSigmoid(gateIH[unit])
					f := lstmSigmoid(gateIH[h+unit])
					g := float32(math.Tanh(float64(gateIH[2*h+unit])))
					o := lstmSigmoid(gateIH[3*h+unit])
					cs[unit] = f*cs[unit] + i*g
					hs[unit] = o * float32(math.Tanh(float64(cs[unit])))
					if math.IsNaN(float64(cs[unit])) || math.IsInf(float64(cs[unit]), 0) {
						return nil, fmt.Errorf("non-finite LSTM cell")
					}
				}
				copy(out[frame*dirs*h+direction*h:frame*dirs*h+(direction+1)*h], hs)
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
		}
		if observe != nil {
			observe(layerIndex, frames, dirs*h, out)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sequence = out
		width = dirs * h
	}
	result.Output = sequence
	return result, nil
}

func lstmSigmoid(x float32) float32 {
	if x >= 0 {
		return float32(1 / (1 + math.Exp(-float64(x))))
	}
	e := math.Exp(float64(x))
	return float32(e / (1 + e))
}
func lstmProjection(ctx context.Context, out, x, w []float32, rows, cols int, mode LSTMMode) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if mode == LSTMSIMD {
		if !simd.GemvRows(out, x, w, rows, cols) {
			return fmt.Errorf("checked LSTM GEMV rejected shape")
		}
	} else {
		for row := 0; row < rows; row++ {
			if row%32 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			var sum float32
			for column, value := range x {
				sum += value * w[row*cols+column]
			}
			out[row] = sum
		}
	}
	return ctx.Err()
}
