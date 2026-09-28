package pockettts

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type CausalConv1D struct {
	Weight, Bias                      []float32
	In, Out, Kernel, Stride, Dilation int
}

type StreamingConvState struct {
	Previous []float32
	First    bool
}

type TransposedConv1D struct {
	Weight, Bias                    []float32
	PhaseFirst, PhaseSecond         []float32
	PackedFirst, PackedSecond       []float32
	In, Out, Kernel, Stride, Groups int
}

type StreamingTransposedState struct{ Partial []float32 }

func (c CausalConv1D) NewState() StreamingConvState {
	return StreamingConvState{Previous: make([]float32, c.In*((c.Kernel-1)*c.Dilation+1-c.Stride)), First: true}
}

func (c TransposedConv1D) NewState() StreamingTransposedState {
	return StreamingTransposedState{Partial: make([]float32, c.Out*(c.Kernel-c.Stride))}
}

func (c CausalConv1D) groups() int {
	if c.In == c.Out && len(c.Weight) == c.Out*c.Kernel {
		return c.Out
	}
	return 1
}

func (c CausalConv1D) outputLength(length int) int {
	if length <= 0 || c.Stride <= 0 || length%c.Stride != 0 {
		return 0
	}
	return length / c.Stride
}

func (c CausalConv1D) scratchFloats(length int) int {
	groups := c.groups()
	prev := (c.Kernel-1)*c.Dilation + 1 - c.Stride
	outLen := c.outputLength(length)
	if groups <= 0 || prev < 0 || outLen <= 0 {
		return 0
	}
	extra := 0
	if outLen >= 16 && (outLen > 256 || (c.In/groups)*c.Kernel > 4096) {
		extra = (c.In / groups) * c.Kernel * 16
	}
	tile := min(outLen, 256)
	return c.In*(prev+length) + (c.In/groups)*c.Kernel*tile + tile + extra
}

// Forward is the allocating convenience API. Warm paths use ForwardInto.
func (c CausalConv1D) Forward(input []float32, length int, state *StreamingConvState) ([]float32, int, error) {
	outLen := c.outputLength(length)
	if outLen == 0 {
		return nil, 0, fmt.Errorf("invalid Pocket TTS causal conv length")
	}
	out := make([]float32, c.Out*outLen)
	scratch, err := NewScratch(c.scratchFloats(length))
	if err != nil {
		return nil, 0, err
	}
	if err := c.ForwardInto(out, input, length, state, scratch); err != nil {
		return nil, 0, err
	}
	return out, outLen, nil
}

// ForwardInto writes channel-major [out,T/stride] into caller-owned output and
// uses caller-owned scratch. It performs no heap allocation when state exists.
func (c CausalConv1D) ForwardInto(out, input []float32, length int, state *StreamingConvState, scratch *Scratch) error {
	outLen := c.outputLength(length)
	groups := c.groups()
	if outLen == 0 || c.In <= 0 || c.Out <= 0 || c.Kernel <= 0 || c.Dilation <= 0 || groups <= 0 || c.In%groups != 0 || c.Out%groups != 0 || len(input) != c.In*length || len(out) < c.Out*outLen || state == nil || scratch == nil {
		return fmt.Errorf("invalid Pocket TTS causal conv")
	}
	inPerGroup := c.In / groups
	effective := (c.Kernel-1)*c.Dilation + 1
	prev := effective - c.Stride
	if prev < 0 || len(state.Previous) != c.In*prev {
		return fmt.Errorf("invalid Pocket TTS causal conv state")
	}
	mark := scratch.Mark()
	joined, err := scratch.Take(c.In * (prev + length))
	if err != nil {
		return err
	}
	total := prev + length
	for ch := 0; ch < c.In; ch++ {
		copy(joined[ch*total:ch*total+prev], state.Previous[ch*prev:(ch+1)*prev])
		copy(joined[ch*total+prev:(ch+1)*total], input[ch*length:(ch+1)*length])
	}
	k := inPerGroup * c.Kernel
	tileCapacity := min(outLen, 256)
	columns, err := scratch.Take(k * tileCapacity)
	if err != nil {
		return err
	}
	ones, err := scratch.Take(tileCapacity)
	if err != nil {
		return err
	}
	for i := range ones {
		ones[i] = 1
	}
	var packedScratch []float32
	if outLen >= 16 && (outLen > 256 || k > 4096) {
		packedScratch, err = scratch.Take(k * 16)
		if err != nil {
			return err
		}
	}
	outPerGroup := c.Out / groups
	for group := 0; group < groups; group++ {
		for tileStart := 0; tileStart < outLen; tileStart += tileCapacity {
			tileWidth := min(tileCapacity, outLen-tileStart)
			tileColumns := columns[:k*tileWidth]
			for icg := 0; icg < inPerGroup; icg++ {
				ic := group*inPerGroup + icg
				for tap := 0; tap < c.Kernel; tap++ {
					row := tileColumns[(icg*c.Kernel+tap)*tileWidth : (icg*c.Kernel+tap+1)*tileWidth]
					for t := 0; t < tileWidth; t++ {
						row[t] = joined[ic*total+(tileStart+t)*c.Stride+tap*c.Dilation]
					}
				}
			}
			for first := 0; first < outPerGroup; {
				count := min(256, outPerGroup-first)
				outputRow := group*outPerGroup + first
				dst := out[outputRow*outLen+tileStart:]
				weights := c.Weight[(group*outPerGroup+first)*k : (group*outPerGroup+first+count)*k]
				if tileWidth < 16 {
					for row := 0; row < count; row++ {
						clear(dst[row*outLen : row*outLen+tileWidth])
					}
					if !simd.SgemmNNTo(dst, weights, tileColumns, count, tileWidth, k, 1, k, tileWidth, outLen) {
						return fmt.Errorf("Pocket TTS causal conv narrow SGEMM failed")
					}
				} else if outLen <= 256 && k <= 4096 {
					contiguous := dst[:count*tileWidth]
					if !simd.FMAMatrixF32Checked(contiguous, weights, tileColumns, count, tileWidth, k) {
						return fmt.Errorf("Pocket TTS causal conv SIMD matrix failed")
					}
				} else if !simd.SgemmNNPackedOverwriteTo(dst, weights, tileColumns, packedScratch, count, tileWidth, k, k, tileWidth, outLen) {
					return fmt.Errorf("Pocket TTS causal conv packed SGEMM failed m=%d n=%d k=%d", count, tileWidth, k)
				}
				if c.Bias != nil {
					for row := 0; row < count; row++ {
						if !simd.VecScaleAddTo(dst[row*outLen:row*outLen+tileWidth], dst[row*outLen:row*outLen+tileWidth], ones[:tileWidth], c.Bias[outputRow+row]) {
							return fmt.Errorf("Pocket TTS causal conv SIMD bias failed")
						}
					}
				}
				first += count
			}
		}
	}
	if prev > 0 {
		for ch := 0; ch < c.In; ch++ {
			copy(state.Previous[ch*prev:(ch+1)*prev], joined[ch*total+total-prev:(ch+1)*total])
		}
	}
	state.First = false
	return scratch.ResetTo(mark)
}

func (c TransposedConv1D) scratchFloats(length int) int {
	if length <= 0 {
		return 0
	}
	if c.Groups == c.In && c.In == c.Out {
		return c.In*length + 2*c.Out
	}
	return c.In*length + 2*length*c.Out + c.Out
}

// Forward is the allocating convenience API. Warm paths use ForwardInto.
func (c TransposedConv1D) Forward(input []float32, length int, state *StreamingTransposedState) ([]float32, int, error) {
	outLen := length * c.Stride
	if outLen <= 0 {
		return nil, 0, fmt.Errorf("invalid Pocket TTS transposed conv length")
	}
	out := make([]float32, c.Out*outLen)
	scratch, err := NewScratch(c.scratchFloats(length))
	if err != nil {
		return nil, 0, err
	}
	local := c
	if len(local.PhaseFirst) == 0 || len(local.PhaseSecond) == 0 {
		local.packPhases()
	}
	if err := local.ForwardInto(out, input, length, state, scratch); err != nil {
		return nil, 0, err
	}
	return out, outLen, nil
}

// ForwardInto writes channel-major [out,T*stride]. Loaded weights must have
// prepacked phases; packing is preparation work, never a warm-path operation.
func (c TransposedConv1D) ForwardInto(out, input []float32, length int, state *StreamingTransposedState, scratch *Scratch) error {
	outLen := length * c.Stride
	if length <= 0 || c.In <= 0 || c.Out <= 0 || c.Kernel != 2*c.Stride || c.Stride <= 0 || c.Groups <= 0 || c.In%c.Groups != 0 || c.Out%c.Groups != 0 || len(input) != c.In*length || len(out) < c.Out*outLen || state == nil || scratch == nil || len(state.Partial) != c.Out*c.Stride || len(c.PhaseFirst) == 0 || len(c.PhaseSecond) == 0 {
		return fmt.Errorf("invalid Pocket TTS transposed conv")
	}
	mark := scratch.Mark()
	timeInput, err := scratch.Take(c.In * length)
	if err != nil {
		return err
	}
	channelToTimePocketInto(timeInput, input, c.In, length)
	if c.Groups == c.In && c.In == c.Out {
		current, err := scratch.Take(c.Out)
		if err != nil {
			return err
		}
		second, err := scratch.Take(c.Out)
		if err != nil {
			return err
		}
		for phase := 0; phase < c.Stride; phase++ {
			w0 := c.PhaseFirst[phase*c.Out : (phase+1)*c.Out]
			w1 := c.PhaseSecond[phase*c.Out : (phase+1)*c.Out]
			for t := 0; t < length; t++ {
				x := timeInput[t*c.In : (t+1)*c.In]
				if !simd.VecMulTo(current, x, w0) {
					return fmt.Errorf("Pocket TTS depthwise transposed SIMD failed")
				}
				if t == 0 {
					for ch := 0; ch < c.Out; ch++ {
						current[ch] += state.Partial[ch*c.Stride+phase]
					}
				} else {
					prev := timeInput[(t-1)*c.In : t*c.In]
					if !simd.VecMulTo(second, prev, w1) || !simd.VecAddTo(current, current, second) {
						return fmt.Errorf("Pocket TTS depthwise overlap SIMD failed")
					}
				}
				if c.Bias != nil && !simd.VecAddTo(current, current, c.Bias) {
					return fmt.Errorf("Pocket TTS depthwise bias SIMD failed")
				}
				for ch := 0; ch < c.Out; ch++ {
					out[ch*outLen+t*c.Stride+phase] = current[ch]
				}
				if t == length-1 {
					if !simd.VecMulTo(second, x, w1) {
						return fmt.Errorf("Pocket TTS depthwise partial SIMD failed")
					}
					for ch := 0; ch < c.Out; ch++ {
						state.Partial[ch*c.Stride+phase] = second[ch]
					}
				}
			}
		}
		return scratch.ResetTo(mark)
	}
	if c.Groups != 1 {
		return fmt.Errorf("unsupported Pocket TTS transposed groups=%d", c.Groups)
	}
	current, err := scratch.Take(length * c.Out)
	if err != nil {
		return err
	}
	second, err := scratch.Take(length * c.Out)
	if err != nil {
		return err
	}
	partial, err := scratch.Take(c.Out)
	if err != nil {
		return err
	}
	for phase := 0; phase < c.Stride; phase++ {
		clear(current)
		clear(second)
		w0 := c.PhaseFirst[phase*c.Out*c.In : (phase+1)*c.Out*c.In]
		w1 := c.PhaseSecond[phase*c.Out*c.In : (phase+1)*c.Out*c.In]
		panelLen := (c.Out / 16) * c.In * 16
		packed0 := c.PackedFirst[phase*panelLen : (phase+1)*panelLen]
		packed1 := c.PackedSecond[phase*panelLen : (phase+1)*panelLen]
		for start := 0; start < length; start += 256 {
			n := min(256, length-start)
			if !simd.SgemmNTPrepackedTo(current[start*c.Out:(start+n)*c.Out], timeInput[start*c.In:(start+n)*c.In], w0, packed0, n, c.Out, c.In, 1, c.In, c.In, c.Out) || !simd.SgemmNTPrepackedTo(second[start*c.Out:(start+n)*c.Out], timeInput[start*c.In:(start+n)*c.In], w1, packed1, n, c.Out, c.In, 1, c.In, c.In, c.Out) {
				return fmt.Errorf("Pocket TTS transposed prepacked GEMM failed")
			}
		}
		for ch := 0; ch < c.Out; ch++ {
			partial[ch] = state.Partial[ch*c.Stride+phase]
		}
		if !simd.VecAddTo(current[:c.Out], current[:c.Out], partial) {
			return fmt.Errorf("Pocket TTS transposed partial add failed")
		}
		if length > 1 && !simd.VecAddTo(current[c.Out:], current[c.Out:], second[:(length-1)*c.Out]) {
			return fmt.Errorf("Pocket TTS transposed overlap add failed")
		}
		if c.Bias != nil && !simd.AddBiasRowsTo(current, c.Bias, length, c.Out) {
			return fmt.Errorf("Pocket TTS transposed bias failed")
		}
		for t := 0; t < length; t++ {
			for ch := 0; ch < c.Out; ch++ {
				out[ch*outLen+t*c.Stride+phase] = current[t*c.Out+ch]
			}
		}
		last := second[(length-1)*c.Out:]
		for ch := 0; ch < c.Out; ch++ {
			state.Partial[ch*c.Stride+phase] = last[ch]
		}
	}
	return scratch.ResetTo(mark)
}

func (c *TransposedConv1D) packPhases() {
	if c.Groups == c.In && c.In == c.Out {
		c.PhaseFirst = make([]float32, c.Stride*c.Out)
		c.PhaseSecond = make([]float32, c.Stride*c.Out)
		for p := 0; p < c.Stride; p++ {
			for ch := 0; ch < c.Out; ch++ {
				c.PhaseFirst[p*c.Out+ch] = c.Weight[ch*c.Kernel+p]
				c.PhaseSecond[p*c.Out+ch] = c.Weight[ch*c.Kernel+p+c.Stride]
			}
		}
		return
	}
	if c.Groups == 1 {
		c.PhaseFirst = make([]float32, c.Stride*c.Out*c.In)
		c.PhaseSecond = make([]float32, c.Stride*c.Out*c.In)
		for p := 0; p < c.Stride; p++ {
			for oc := 0; oc < c.Out; oc++ {
				for ic := 0; ic < c.In; ic++ {
					src := (ic*c.Out + oc) * c.Kernel
					c.PhaseFirst[(p*c.Out+oc)*c.In+ic] = c.Weight[src+p]
					c.PhaseSecond[(p*c.Out+oc)*c.In+ic] = c.Weight[src+p+c.Stride]
				}
			}
		}
		panelLen := (c.Out / 16) * c.In * 16
		c.PackedFirst = make([]float32, c.Stride*panelLen)
		c.PackedSecond = make([]float32, c.Stride*panelLen)
		for p := 0; p < c.Stride; p++ {
			raw0 := c.PhaseFirst[p*c.Out*c.In : (p+1)*c.Out*c.In]
			raw1 := c.PhaseSecond[p*c.Out*c.In : (p+1)*c.Out*c.In]
			packed0, err0 := simd.PackSgemmNTWeightsInto(raw0, c.Out, c.In, c.In, c.PackedFirst[p*panelLen:(p+1)*panelLen])
			packed1, err1 := simd.PackSgemmNTWeightsInto(raw1, c.Out, c.In, c.In, c.PackedSecond[p*panelLen:(p+1)*panelLen])
			if err0 != nil || err1 != nil || len(packed0) != panelLen || len(packed1) != panelLen {
				c.PackedFirst = nil
				c.PackedSecond = nil
				return
			}
		}
	}
}

func channelToTimePocketInto(out, x []float32, channels, length int) {
	for ch := 0; ch < channels; ch++ {
		for t := 0; t < length; t++ {
			out[t*channels+ch] = x[ch*length+t]
		}
	}
}
