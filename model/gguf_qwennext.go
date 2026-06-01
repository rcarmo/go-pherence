package model

import (
	"fmt"
	"math"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func (c GGUFLlamaConfig) IsQwenNextHybridGGUF() bool {
	return c.SSMInnerSize > 0 || c.SSMStateSize > 0 || c.FullAttentionInterval > 0 || c.AttentionKeyLength > 0 || c.AttentionValueLength > 0
}

func (c GGUFLlamaConfig) ValidateQwenNextHybridMetadata() error {
	if !c.IsQwenNextHybridGGUF() {
		return nil
	}
	if c.HiddenSize <= 0 || c.SSMInnerSize <= 0 || c.SSMConvKernel <= 0 || c.SSMGroupCount <= 0 || c.SSMStateSize <= 0 || c.SSMTimeStepRank <= 0 {
		return fmt.Errorf("incomplete QwenNext GGUF SSM metadata hidden=%d inner=%d kernel=%d groups=%d state=%d rank=%d", c.HiddenSize, c.SSMInnerSize, c.SSMConvKernel, c.SSMGroupCount, c.SSMStateSize, c.SSMTimeStepRank)
	}
	if c.AttentionKeyLength <= 0 || c.AttentionValueLength <= 0 || c.FullAttentionInterval <= 0 {
		return fmt.Errorf("incomplete QwenNext GGUF attention metadata key=%d value=%d interval=%d", c.AttentionKeyLength, c.AttentionValueLength, c.FullAttentionInterval)
	}
	return nil
}

func loadGGUFQwenNextHybridTensors(g *gguf.GGUF, layerIdx int, layer *GGUFLlamaLayer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("nil GGUF QwenNext load input")
	}
	p := fmt.Sprintf("blk.%d.", layerIdx)
	loadMatrixOptional := func(name string) (*gguf.QuantMatrix, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, nil
		}
		return g.MatrixFromTensor(t)
	}
	loadFloatOptional := func(name string) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, nil
		}
		return g.DequantF32(t)
	}
	var err error
	if layer.FusedQKVM, err = loadMatrixOptional(p + "attn_qkv.weight"); err != nil {
		return err
	}
	if layer.AttnGateM, err = loadMatrixOptional(p + "attn_gate.weight"); err != nil {
		return err
	}
	if layer.SSMOutM, err = loadMatrixOptional(p + "ssm_out.weight"); err != nil {
		return err
	}
	if layer.SSMAlphaM, err = loadMatrixOptional(p + "ssm_alpha.weight"); err != nil {
		return err
	}
	if layer.SSMBetaM, err = loadMatrixOptional(p + "ssm_beta.weight"); err != nil {
		return err
	}
	if layer.SSMConv1D, err = loadFloatOptional(p + "ssm_conv1d.weight"); err != nil {
		return err
	}
	if layer.SSMA, err = loadFloatOptional(p + "ssm_a"); err != nil {
		return err
	}
	if layer.SSMDTBias, err = loadFloatOptional(p + "ssm_dt.bias"); err != nil {
		return err
	}
	if layer.SSMNorm, err = loadFloatOptional(p + "ssm_norm.weight"); err != nil {
		return err
	}
	return nil
}

func (c GGUFLlamaConfig) QwenNextLinearShapes() (loaderconfig.Qwen35LinearAttentionShapes, error) {
	return loaderconfig.Qwen35LinearAttentionShapesFor(c.HiddenSize, c.SSMInnerSize, c.SSMStateSize, c.SSMConvKernel, c.SSMTimeStepRank, c.SSMGroupCount)
}

func (l GGUFLlamaLayer) HasQwenNextHybridTensors() bool {
	return l.FusedQKVM != nil || l.SSMOutM != nil || l.SSMConv1D != nil || l.SSMA != nil
}

type GGUFQwenNextState struct {
	Conv []float32
	SSM  []float32
	Pos  int
}

func (c GGUFLlamaConfig) NewQwenNextState() (GGUFQwenNextState, error) {
	shapes, err := c.QwenNextLinearShapes()
	if err != nil {
		return GGUFQwenNextState{}, err
	}
	ssmLen := c.SSMTimeStepRank * c.SSMInnerSize * c.SSMStateSize
	if ssmLen < 0 {
		return GGUFQwenNextState{}, fmt.Errorf("QwenNext SSM state overflow")
	}
	return GGUFQwenNextState{Conv: make([]float32, shapes.ConvDim*c.SSMConvKernel), SSM: make([]float32, ssmLen)}, nil
}

func (m *GGUFLlama) projectQwenNextFusedQKV(layer *GGUFLlamaLayer, x []float32) (q, k, v []float32, err error) {
	if m == nil || layer == nil || layer.FusedQKVM == nil {
		return nil, nil, nil, fmt.Errorf("missing QwenNext fused qkv tensor")
	}
	shapes, err := m.Config.QwenNextLinearShapes()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(x) != m.Config.HiddenSize {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv input len=%d want %d", len(x), m.Config.HiddenSize)
	}
	if layer.FusedQKVM.InDim != m.Config.HiddenSize || layer.FusedQKVM.OutDim != shapes.ConvDim {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv dims in/out=%d/%d want %d/%d", layer.FusedQKVM.InDim, layer.FusedQKVM.OutDim, m.Config.HiddenSize, shapes.ConvDim)
	}
	projected := make([]float32, shapes.ConvDim)
	m.gemvMaybe(projected, x, nil, layer.FusedQKVM, m.Config.HiddenSize, shapes.ConvDim)
	return splitGGUFQwenNextFusedQKV(projected, shapes)
}

func (m *GGUFLlama) projectQwenNextGate(layer *GGUFLlamaLayer, x []float32) ([]float32, error) {
	if m == nil || layer == nil || layer.AttnGateM == nil {
		return nil, fmt.Errorf("missing QwenNext attention gate tensor")
	}
	shapes, err := m.Config.QwenNextLinearShapes()
	if err != nil {
		return nil, err
	}
	z := make([]float32, shapes.ValueDim)
	m.gemvMaybe(z, x, nil, layer.AttnGateM, m.Config.HiddenSize, shapes.ValueDim)
	return z, nil
}

func splitGGUFQwenNextFusedQKV(projected []float32, shapes loaderconfig.Qwen35LinearAttentionShapes) (q, k, v []float32, err error) {
	if len(projected) != shapes.ConvDim || shapes.KeyDim <= 0 || shapes.ValueDim <= 0 || shapes.ConvDim != shapes.KeyDim*2+shapes.ValueDim {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv len=%d want conv=%d key=%d value=%d", len(projected), shapes.ConvDim, shapes.KeyDim, shapes.ValueDim)
	}
	q = append([]float32(nil), projected[:shapes.KeyDim]...)
	k = append([]float32(nil), projected[shapes.KeyDim:2*shapes.KeyDim]...)
	v = append([]float32(nil), projected[2*shapes.KeyDim:]...)
	return q, k, v, nil
}

func updateGGUFQwenNextConvStateInPlace(state, q, k, v []float32, kernel int) error {
	if kernel <= 0 {
		return fmt.Errorf("invalid QwenNext conv kernel %d", kernel)
	}
	convDim := len(q) + len(k) + len(v)
	if convDim <= 0 || len(state) != convDim*kernel {
		return fmt.Errorf("QwenNext conv state len=%d want %d", len(state), convDim*kernel)
	}
	copy(state, state[convDim:])
	pos := (kernel - 1) * convDim
	copy(state[pos:pos+len(q)], q)
	copy(state[pos+len(q):pos+len(q)+len(k)], k)
	copy(state[pos+len(q)+len(k):pos+convDim], v)
	return nil
}

func applyGGUFQwenNextDepthwiseConv(state, weight []float32, convDim, kernel int) ([]float32, error) {
	if convDim <= 0 || kernel <= 0 || len(state) != convDim*kernel || len(weight) != convDim*kernel {
		return nil, fmt.Errorf("QwenNext conv dims state=%d weight=%d want %d", len(state), len(weight), convDim*kernel)
	}
	out := make([]float32, convDim)
	for c := 0; c < convDim; c++ {
		var sum float32
		for k := 0; k < kernel; k++ {
			sum += state[k*convDim+c] * weight[k*convDim+c]
		}
		out[c] = sum
	}
	return out, nil
}

func prepareGGUFQwenNextDeltaParams(alpha, beta, dtBias, a []float32, rank int) (dt, decay []float32, err error) {
	if rank <= 0 || len(alpha) != rank || len(beta) != rank || len(dtBias) != rank || len(a) != rank {
		return nil, nil, fmt.Errorf("QwenNext delta lens alpha/beta/dt_bias/a=%d/%d/%d/%d want %d", len(alpha), len(beta), len(dtBias), len(a), rank)
	}
	dt = make([]float32, rank)
	decay = make([]float32, rank)
	for i := 0; i < rank; i++ {
		dt[i] = ggufQwenNextSoftplus(alpha[i] + dtBias[i])
		decay[i] = float32(math.Exp(float64(dt[i] * a[i])))
		beta[i] = ggufQwenNextSigmoid(beta[i])
	}
	return dt, decay, nil
}

func applyGGUFQwenNextDeltaUpdateInPlace(ssm, q, k, v, beta, dt, decay []float32, cfg GGUFLlamaConfig) ([]float32, error) {
	shapes, err := cfg.QwenNextLinearShapes()
	if err != nil {
		return nil, err
	}
	keyDim := cfg.SSMGroupCount * cfg.SSMStateSize
	valueHeads := cfg.SSMTimeStepRank
	valueHeadDim := cfg.SSMInnerSize / valueHeads
	stateLen := valueHeads * valueHeadDim * cfg.SSMStateSize
	if len(ssm) != stateLen || len(q) != keyDim || len(k) != keyDim || len(v) != shapes.ValueDim || len(beta) != valueHeads || len(dt) != valueHeads || len(decay) != valueHeads {
		return nil, fmt.Errorf("QwenNext delta dims ssm/q/k/v/beta/dt/decay=%d/%d/%d/%d/%d/%d/%d want %d/%d/%d/%d/%d", len(ssm), len(q), len(k), len(v), len(beta), len(dt), len(decay), stateLen, keyDim, keyDim, shapes.ValueDim, valueHeads)
	}
	if valueHeads%cfg.SSMGroupCount != 0 {
		return nil, fmt.Errorf("QwenNext value heads %d not divisible by key groups %d", valueHeads, cfg.SSMGroupCount)
	}
	out := make([]float32, shapes.ValueDim)
	rep := valueHeads / cfg.SSMGroupCount
	scale := float32(1.0 / math.Sqrt(float64(cfg.SSMStateSize)))
	for vh := 0; vh < valueHeads; vh++ {
		keyHead := vh / rep
		qHead := q[keyHead*cfg.SSMStateSize : (keyHead+1)*cfg.SSMStateSize]
		kHead := k[keyHead*cfg.SSMStateSize : (keyHead+1)*cfg.SSMStateSize]
		stateHeadBase := vh * valueHeadDim * cfg.SSMStateSize
		valueBase := vh * valueHeadDim
		for vd := 0; vd < valueHeadDim; vd++ {
			idx := valueBase + vd
			row := ssm[stateHeadBase+vd*cfg.SSMStateSize : stateHeadBase+(vd+1)*cfg.SSMStateSize]
			out[idx] = ggufQwenNextDeltaRowInPlace(row, qHead, kHead, v[idx], beta[vh], decay[vh], scale)
		}
	}
	return out, nil
}

func ggufQwenNextDeltaRowInPlace(stateRow, qHead, kHead []float32, vVal, betaV, decayV, scale float32) float32 {
	var acc float32
	for i := range kHead {
		old := stateRow[i]
		updated := old*decayV + betaV*(vVal-old*kHead[i])*kHead[i]
		stateRow[i] = updated
		acc += updated * qHead[i]
	}
	return acc * scale
}

func applyGGUFQwenNextGatedRMSNormValueHeads(x, gate, weight []float32, heads, headDim int, eps float32) error {
	if heads <= 0 || headDim <= 0 || len(x) != heads*headDim || len(gate) != len(x) || len(weight) != headDim {
		return fmt.Errorf("QwenNext gated RMSNorm dims x/gate/weight=%d/%d/%d heads=%d head_dim=%d", len(x), len(gate), len(weight), heads, headDim)
	}
	for h := 0; h < heads; h++ {
		row := x[h*headDim : (h+1)*headDim]
		var ss float32
		for _, v := range row {
			ss += v * v
		}
		scale := float32(1.0 / math.Sqrt(float64(ss/float32(headDim)+eps)))
		for i := 0; i < headDim; i++ {
			g := gate[h*headDim+i]
			row[i] = row[i] * scale * weight[i] * g * ggufQwenNextSigmoid(g)
		}
	}
	return nil
}

func ggufQwenNextSoftplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func ggufQwenNextSigmoid(x float32) float32 {
	if x >= 0 {
		e := float32(math.Exp(float64(-x)))
		return 1 / (1 + e)
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}
func (m *GGUFLlama) forwardQwenNextHybridLayer(out []float32, layer *GGUFLlamaLayer, state *GGUFQwenNextState, input []float32) error {
	if m == nil || layer == nil || state == nil {
		return fmt.Errorf("nil QwenNext hybrid forward input")
	}
	cfg := m.Config
	shapes, err := cfg.QwenNextLinearShapes()
	if err != nil {
		return err
	}
	if len(input) != cfg.HiddenSize || len(out) < cfg.HiddenSize {
		return fmt.Errorf("QwenNext hybrid input/output len=%d/%d want %d", len(input), len(out), cfg.HiddenSize)
	}
	q, k, v, err := m.projectQwenNextFusedQKV(layer, input)
	if err != nil {
		return err
	}
	z, err := m.projectQwenNextGate(layer, input)
	if err != nil {
		return err
	}
	if err := updateGGUFQwenNextConvStateInPlace(state.Conv, q, k, v, cfg.SSMConvKernel); err != nil {
		return err
	}
	convOut, err := applyGGUFQwenNextDepthwiseConv(state.Conv, layer.SSMConv1D, shapes.ConvDim, cfg.SSMConvKernel)
	if err != nil {
		return err
	}
	ggufQwenNextSiluInPlace(convOut)
	q, k, v, err = splitGGUFQwenNextFusedQKV(convOut, shapes)
	if err != nil {
		return err
	}
	alpha := make([]float32, cfg.SSMTimeStepRank)
	beta := make([]float32, cfg.SSMTimeStepRank)
	m.gemvMaybe(alpha, input, nil, layer.SSMAlphaM, cfg.HiddenSize, cfg.SSMTimeStepRank)
	m.gemvMaybe(beta, input, nil, layer.SSMBetaM, cfg.HiddenSize, cfg.SSMTimeStepRank)
	dt, decay, err := prepareGGUFQwenNextDeltaParams(alpha, beta, layer.SSMDTBias, layer.SSMA, cfg.SSMTimeStepRank)
	if err != nil {
		return err
	}
	deltaOut, err := applyGGUFQwenNextDeltaUpdateInPlace(state.SSM, q, k, v, beta, dt, decay, cfg)
	if err != nil {
		return err
	}
	if err := applyGGUFQwenNextGatedRMSNormValueHeads(deltaOut, z, layer.SSMNorm, cfg.SSMTimeStepRank, cfg.SSMInnerSize/cfg.SSMTimeStepRank, cfg.RMSNormEps); err != nil {
		return err
	}
	m.gemvMaybe(out, deltaOut, nil, layer.SSMOutM, shapes.ValueDim, cfg.HiddenSize)
	state.Pos++
	return nil
}

func ggufQwenNextSiluInPlace(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}
