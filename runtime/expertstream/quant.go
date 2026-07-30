package expertstream

import "fmt"

// DTypeMLXQuant is the canonical ComponentSpec.DType value for a component
// carrying an MLX affine-quantized sub-layout (see QuantSpec). A component
// must set exactly one of {DType == DTypeMLXQuant, Quant != nil} together;
// validateManifest rejects any component where the two disagree.
const DTypeMLXQuant = "mlxq"

// QuantLayout holds validated byte sizes for one quantized component's
// sub-regions, in the fixed order weight || scales || biases.
type QuantLayout struct {
	WeightSize int64
	ScaleSize  int64
	BiasSize   int64
	Groups     int64
	PackFactor int64
}

// Layout validates q in isolation and against componentSize, returning the
// byte sub-ranges that must exactly tile componentSize with no gaps or
// overlap. It never allocates or touches component bytes.
func (q *QuantSpec) Layout(componentSize int64) (QuantLayout, error) {
	if q == nil {
		return QuantLayout{}, fmt.Errorf("mlx quant spec is nil")
	}
	if componentSize <= 0 {
		return QuantLayout{}, fmt.Errorf("mlx quant component size=%d must be positive", componentSize)
	}
	if q.OutDim <= 0 || q.InDim <= 0 || q.GroupSize <= 0 {
		return QuantLayout{}, fmt.Errorf("mlx quant dims must be positive out_dim=%d in_dim=%d group_size=%d", q.OutDim, q.InDim, q.GroupSize)
	}
	if q.Bits <= 0 || q.Bits > 32 || 32%q.Bits != 0 {
		return QuantLayout{}, fmt.Errorf("mlx quant bits=%d must be a positive divisor of 32", q.Bits)
	}
	packFactor := int64(32 / q.Bits)
	if q.InDim%packFactor != 0 {
		return QuantLayout{}, fmt.Errorf("mlx quant in_dim=%d is not divisible by pack_factor=%d (bits=%d)", q.InDim, packFactor, q.Bits)
	}
	if q.InDim%q.GroupSize != 0 {
		return QuantLayout{}, fmt.Errorf("mlx quant in_dim=%d is not divisible by group_size=%d", q.InDim, q.GroupSize)
	}
	groups := q.InDim / q.GroupSize

	weightElems, err := checkedProduct(q.OutDim, q.InDim/packFactor)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant weight element count overflows")
	}
	weightSize, err := checkedProduct(weightElems, 4)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant weight byte size overflows")
	}
	scaleElems, err := checkedProduct(q.OutDim, groups)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant scale element count overflows")
	}
	scaleSize, err := checkedProduct(scaleElems, 4)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant scale byte size overflows")
	}
	biasSize := scaleSize

	total, err := checkedAdd(weightSize, scaleSize)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant total byte size overflows")
	}
	total, err = checkedAdd(total, biasSize)
	if err != nil {
		return QuantLayout{}, fmt.Errorf("mlx quant total byte size overflows")
	}
	if total != componentSize {
		return QuantLayout{}, fmt.Errorf("mlx quant total size=%d does not match component size=%d (weight=%d scale=%d bias=%d)", total, componentSize, weightSize, scaleSize, biasSize)
	}
	return QuantLayout{WeightSize: weightSize, ScaleSize: scaleSize, BiasSize: biasSize, Groups: groups, PackFactor: packFactor}, nil
}

// SplitBytes validates q against len(raw) and splits raw (a component's own
// byte slice, as returned via Component.Bytes) into weight/scale/bias byte
// sub-slices, in that fixed order. The returned slices alias raw; no copies
// are made.
func (q *QuantSpec) SplitBytes(raw []byte) (weight, scales, biases []byte, err error) {
	layout, err := q.Layout(int64(len(raw)))
	if err != nil {
		return nil, nil, nil, err
	}
	w := raw[:layout.WeightSize]
	s := raw[layout.WeightSize : layout.WeightSize+layout.ScaleSize]
	b := raw[layout.WeightSize+layout.ScaleSize:]
	return w, s, b, nil
}

func cloneQuantSpec(in *QuantSpec) *QuantSpec {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}
