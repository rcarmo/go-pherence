//go:build ggml && cgo && linux

package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGemma4GGMLFlashAttentionF16Oracle(t *testing.T) {
	assertFlashAttentionF16Oracle(t, "compact", 8, 5, 4, 2, 0.113, -0.071, 0.091, 5e-4)
}

func TestGemma4GGMLFlashAttentionF16OracleGemma4Sized(t *testing.T) {
	assertFlashAttentionF16Oracle(t, "gemma4-sized", 256, 5, 8, 2, 0.011, -0.007, 0.009, 7e-4)
}

func assertFlashAttentionF16Oracle(t *testing.T, name string, headDim, seqLen, nHead, nKV int, qScale, kScale, vScale float64, tol float64) {
	t.Helper()
	q := syntheticOracleVec(headDim*nHead, qScale)
	k := syntheticOracleVec(headDim*seqLen*nKV, kScale)
	v := syntheticOracleVec(headDim*seqLen*nKV, vScale)
	kF16 := make([]uint16, len(k))
	vF16 := make([]uint16, len(v))
	kRounded := make([]float32, len(k))
	vRounded := make([]float32, len(v))
	// Go attention cache layout is [seq][kv_head][dim]. ggml tensor layout for
	// flash attention is [dim][seq][kv_head] with dim contiguous, i.e.
	// [kv_head][seq][dim] in flat row-major storage.
	for seq := 0; seq < seqLen; seq++ {
		for kvh := 0; kvh < nKV; kvh++ {
			for d := 0; d < headDim; d++ {
				goIdx := seq*nKV*headDim + kvh*headDim + d
				ggmlIdx := kvh*seqLen*headDim + seq*headDim + d
				kF16[ggmlIdx] = half.F32ToF16(k[goIdx])
				vF16[ggmlIdx] = half.F32ToF16(v[goIdx])
				kRounded[goIdx] = half.F16ToF32(kF16[ggmlIdx])
				vRounded[goIdx] = half.F16ToF32(vF16[ggmlIdx])
			}
		}
	}
	gotGGML := make([]float32, headDim*nHead)
	if err := ggmlcompute.FlashAttnF32F16(gotGGML, q, kF16, vF16, nil, headDim, seqLen, nHead, nKV, 1.0); err != nil {
		t.Fatal(err)
	}
	gotGoF32Accum := gqaAttentionScale(q, kRounded, vRounded, seqLen, nHead, nKV, headDim, 1.0)
	maxF32, meanF32 := maxMeanAbsDiff(gotGGML, gotGoF32Accum)
	gotGoF16Accum := flashAttnF16AccumReference(q, kRounded, vRounded, seqLen, nHead, nKV, headDim, 1.0)
	maxF16, meanF16 := maxMeanAbsDiff(gotGGML, gotGoF16Accum)
	t.Logf("%s ggml flash_attn_ext F16 KV vs Go F32-accum max=%g mean=%g; F16-accum max=%g mean=%g", name, maxF32, meanF32, maxF16, meanF16)
	if maxF16 > tol {
		t.Fatalf("%s ggml flash_attn_ext F16 KV vs F16-accum reference max=%g mean=%g tol=%g", name, maxF16, meanF16, tol)
	}
	if maxF32 <= maxF16 {
		t.Fatalf("%s F32 accumulation unexpectedly no farther from ggml than F16 accumulation: f32=%g f16=%g", name, maxF32, maxF16)
	}
}

func TestGemma4RealGGMLRMSNormOracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		w    []float32
		x    []float32
	}{
		{name: "blk.0.attn_norm", w: m.Layers[0].InputNorm.Data(), x: syntheticOracleVec(m.Config.HiddenSize, 0.011)},
		{name: "blk.0.post_attention_norm", w: m.Layers[0].PostNorm.Data(), x: syntheticOracleVec(m.Config.HiddenSize, -0.013)},
		{name: "blk.0.ffn_norm", w: m.Layers[0].PreFFNNorm.Data(), x: syntheticOracleVec(m.Config.HiddenSize, 0.017)},
		{name: "blk.0.post_ffw_norm", w: m.Layers[0].PostFFNNorm.Data(), x: syntheticOracleVec(m.Config.HiddenSize, -0.019)},
		{name: "output_norm", w: m.Norm.Data(), x: syntheticOracleVec(m.Config.HiddenSize, 0.023)},
	}
	for _, tc := range cases {
		gotGGML := make([]float32, len(tc.x))
		if err := ggmlcompute.RMSNormMulF32(gotGGML, tc.x, tc.w, float32(m.Config.RMSNormEps)); err != nil {
			t.Fatalf("ggml %s rms_norm: %v", tc.name, err)
		}
		gotGo := append([]float32(nil), tc.x...)
		rmsNormInPlace(gotGo, tc.w, float32(m.Config.RMSNormEps))
		maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
		t.Logf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		if maxDiff > 2e-6 {
			t.Fatalf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		}
	}
}

func TestGemma4RealGGMLRMSNormNoScaleOracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	headDim := m.Layers[0].HeadDimLocal
	x := syntheticOracleVec(headDim, 0.071)
	ones := make([]float32, headDim)
	for i := range ones {
		ones[i] = 1
	}
	gotGGML := make([]float32, headDim)
	if err := ggmlcompute.RMSNormMulF32(gotGGML, x, ones, float32(m.Config.RMSNormEps)); err != nil {
		t.Fatal(err)
	}
	gotGo := append([]float32(nil), x...)
	simd.RMSNormNoScale(gotGo, float32(m.Config.RMSNormEps))
	maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
	t.Logf("Gemma4 no-scale RMSNorm ggml-vs-go max=%g mean=%g", maxDiff, meanDiff)
	if maxDiff > 2e-6 {
		t.Fatalf("Gemma4 no-scale RMSNorm ggml-vs-go max=%g mean=%g", maxDiff, meanDiff)
	}
}

func TestGemma4GGMLGEGLUSplitOracle(t *testing.T) {
	gate := syntheticOracleVec(257, 0.083)
	up := syntheticOracleVec(257, -0.047)
	gotGGML := make([]float32, len(gate))
	if err := ggmlcompute.GEGLUSplitF32(gotGGML, gate, up); err != nil {
		t.Fatal(err)
	}
	gotGo := append([]float32(nil), gate...)
	ggmlGELUMulInPlace(gotGo, up)
	maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
	t.Logf("ggml_geglu_split-vs-go max=%g mean=%g", maxDiff, meanDiff)
	if maxDiff > 1e-6 {
		t.Fatalf("ggml_geglu_split-vs-go max=%g mean=%g", maxDiff, meanDiff)
	}
}

func TestGemma4RepresentativeLayerProjectionRealGGMLQ4Oracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	layers := []int{0, 6, 22, 23, 24, 41}
	for _, layerIdx := range layers {
		if layerIdx < 0 || layerIdx >= len(m.Layers) {
			t.Fatalf("layer %d outside model layers=%d", layerIdx, len(m.Layers))
		}
		l := m.Layers[layerIdx]
		cases := []struct {
			name string
			mat  *gguf.QuantMatrix
			in   []float32
		}{
			{name: "attn_q.weight", mat: l.QWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.017+float64(layerIdx)*0.001)},
			{name: "attn_output.weight", mat: l.OWGGUF, in: syntheticOracleVec(m.Config.NumHeads*l.HeadDimLocal, -0.029-float64(layerIdx)*0.001)},
			{name: "ffn_gate.weight", mat: l.GateWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.031+float64(layerIdx)*0.001)},
			{name: "ffn_up.weight", mat: l.UpWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, -0.037-float64(layerIdx)*0.001)},
			{name: "ffn_down.weight", mat: l.DownWGGUF, in: syntheticOracleVec(l.DownWGGUF.InDim, 0.043+float64(layerIdx)*0.001)},
		}
		if l.HasKV {
			cases = append(cases,
				struct {
					name string
					mat  *gguf.QuantMatrix
					in   []float32
				}{name: "attn_k.weight", mat: l.KWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.019+float64(layerIdx)*0.001)},
				struct {
					name string
					mat  *gguf.QuantMatrix
					in   []float32
				}{name: "attn_v.weight", mat: l.VWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.023+float64(layerIdx)*0.001)},
			)
		}
		for _, tc := range cases {
			if tc.mat == nil {
				t.Fatalf("layer %d %s not loaded", layerIdx, tc.name)
			}
			fullName := "blk." + itoa(layerIdx) + "." + tc.name
			tensor, ok := g.TensorByName(fullName)
			if !ok {
				t.Fatalf("%s not found", fullName)
			}
			raw, err := g.Raw(tensor)
			if err != nil {
				t.Fatal(err)
			}
			gotGGML := make([]float32, tc.mat.OutDim)
			if err := ggmlcompute.MulMatQuantF32(int(tensor.QType), gotGGML, raw, tc.in, tc.mat.InDim, tc.mat.OutDim); err != nil {
				t.Fatalf("ggml %s mul_mat: %v", fullName, err)
			}
			gotGo := make([]float32, tc.mat.OutDim)
			if !gemvGGUFTo(gotGo, tc.in, tc.mat, tc.mat.InDim, tc.mat.OutDim) {
				t.Fatalf("go %s gemv rejected", fullName)
			}
			maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
			t.Logf("%s ggml-vs-go max=%g mean=%g", fullName, maxDiff, meanDiff)
			if maxDiff > 1e-4 {
				t.Fatalf("%s ggml-vs-go max=%g mean=%g", fullName, maxDiff, meanDiff)
			}
		}
	}
}

func TestGemma4Layer0ProjectionRealGGMLQ4Oracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	l0 := m.Layers[0]
	cases := []struct {
		name string
		mat  *gguf.QuantMatrix
		in   []float32
	}{
		{name: "blk.0.attn_q.weight", mat: l0.QWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.017)},
		{name: "blk.0.attn_k.weight", mat: l0.KWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.019)},
		{name: "blk.0.attn_v.weight", mat: l0.VWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.023)},
		{name: "blk.0.attn_output.weight", mat: l0.OWGGUF, in: syntheticOracleVec(m.Config.NumHeads*l0.HeadDimLocal, -0.029)},
		{name: "blk.0.ffn_gate.weight", mat: l0.GateWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.031)},
		{name: "blk.0.ffn_up.weight", mat: l0.UpWGGUF, in: syntheticOracleVec(m.Config.HiddenSize, -0.037)},
		{name: "blk.0.ffn_down.weight", mat: l0.DownWGGUF, in: syntheticOracleVec(l0.DownWGGUF.InDim, 0.043)},
	}
	for _, tc := range cases {
		if tc.mat == nil {
			t.Fatalf("%s not loaded", tc.name)
		}
		tensor, ok := g.TensorByName(tc.name)
		if !ok {
			t.Fatalf("%s not found", tc.name)
		}
		raw, err := g.Raw(tensor)
		if err != nil {
			t.Fatal(err)
		}
		gotGGML := make([]float32, tc.mat.OutDim)
		if err := ggmlcompute.MulMatQuantF32(int(tensor.QType), gotGGML, raw, tc.in, tc.mat.InDim, tc.mat.OutDim); err != nil {
			t.Fatalf("ggml %s mul_mat: %v", tc.name, err)
		}
		gotGo := make([]float32, tc.mat.OutDim)
		if !gemvGGUFTo(gotGo, tc.in, tc.mat, tc.mat.InDim, tc.mat.OutDim) {
			t.Fatalf("go %s gemv rejected", tc.name)
		}
		maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
		t.Logf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		if maxDiff > 1e-4 {
			t.Fatalf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		}
	}
}

func TestGemma4PLIGateProjectionRealGGMLQ4Oracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cases := []struct {
		name string
		mat  *gguf.QuantMatrix
		in   []float32
	}{
		{name: "blk.0.inp_gate.weight", mat: m.Layers[0].PLIGateGGUF, in: syntheticOracleVec(m.Config.HiddenSize, 0.019)},
		{name: "blk.0.proj.weight", mat: m.Layers[0].PLIProjGGUF, in: syntheticOracleVec(m.Config.HiddenPerLayer, -0.041)},
	}
	for _, tc := range cases {
		if tc.mat == nil {
			t.Fatalf("%s not loaded", tc.name)
		}
		tensor, ok := g.TensorByName(tc.name)
		if !ok {
			t.Fatalf("%s not found", tc.name)
		}
		raw, err := g.Raw(tensor)
		if err != nil {
			t.Fatal(err)
		}
		gotGGML := make([]float32, tc.mat.OutDim)
		if err := ggmlcompute.MulMatQuantF32(int(tensor.QType), gotGGML, raw, tc.in, tc.mat.InDim, tc.mat.OutDim); err != nil {
			t.Fatalf("ggml %s mul_mat: %v", tc.name, err)
		}
		gotGo := make([]float32, tc.mat.OutDim)
		if !gemvGGUFTo(gotGo, tc.in, tc.mat, tc.mat.InDim, tc.mat.OutDim) {
			t.Fatalf("go %s gemv rejected", tc.name)
		}
		maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
		t.Logf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		if maxDiff > 1e-4 {
			t.Fatalf("%s ggml-vs-go max=%g mean=%g", tc.name, maxDiff, meanDiff)
		}
	}
}

func TestGemma4PerLayerTokenEmbeddingRealGGMLGetRowsOracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	tensor, ok := g.TensorByName("per_layer_token_embd.weight")
	if !ok {
		t.Fatal("per_layer_token_embd.weight not found")
	}
	raw, err := g.Raw(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if m.EmbedPerLayerGGUF == nil {
		t.Fatal("Go per-layer GGUF embedding not loaded")
	}
	probeTokens := []int{10979, 236764, 564, 236789}
	for _, tok := range probeTokens {
		gotGGML := make([]float32, m.EmbedPerLayerGGUF.InDim)
		if err := ggmlcompute.GetRowToF32(int(tensor.QType), gotGGML, raw, m.EmbedPerLayerGGUF.InDim, m.EmbedPerLayerGGUF.OutDim, tok); err != nil {
			t.Fatalf("ggml get row token %d: %v", tok, err)
		}
		gotGo := make([]float32, m.EmbedPerLayerGGUF.InDim)
		if err := m.EmbedPerLayerGGUF.DequantRowTo(gotGo, tok); err != nil {
			t.Fatalf("go dequant row token %d: %v", tok, err)
		}
		maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotGo)
		t.Logf("per_layer_token_embd token=%d ggml-vs-go max=%g mean=%g", tok, maxDiff, meanDiff)
		if maxDiff > 1e-6 {
			t.Fatalf("per_layer_token_embd token=%d ggml-vs-go max=%g mean=%g", tok, maxDiff, meanDiff)
		}
	}
}

func TestGemma4PerLayerModelProjRealGGMLF16Oracle(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join("models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
		if root := findMTPParityRepoRoot(); root != "" {
			path = filepath.Join(root, path)
		}
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("Gemma4 GGUF not available at %s", path)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	tensor, ok := g.TensorByName("per_layer_model_proj.weight")
	if !ok {
		t.Fatal("per_layer_model_proj.weight not found")
	}
	if tensor.QType != gguf.QuantF16 {
		t.Fatalf("per_layer_model_proj.weight type=%s, want F16", tensor.QType)
	}
	if len(tensor.Shape) != 2 || int(tensor.Shape[0]) != m.Config.HiddenSize || int(tensor.Shape[1]) != m.Config.NumLayers*m.Config.HiddenPerLayer {
		t.Fatalf("per_layer_model_proj.weight shape=%v does not match hidden/layers/hpl=%d/%d/%d", tensor.Shape, m.Config.HiddenSize, m.Config.NumLayers, m.Config.HiddenPerLayer)
	}
	raw, err := g.Raw(tensor)
	if err != nil {
		t.Fatal(err)
	}
	wF16 := make([]uint16, len(raw)/2)
	for i := range wF16 {
		wF16[i] = binary.LittleEndian.Uint16(raw[i*2:])
	}

	fx := []int{10979, 236764}
	if fixturePath := os.Getenv("GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE"); fixturePath != "" {
		_ = fixturePath // this test only needs the canonical first prompt token, kept for env discoverability
	}
	hidden := make([]float32, m.Config.HiddenSize)
	if err := m.ScaledTokenEmbeddingInto(hidden, fx[0]); err != nil {
		t.Fatal(err)
	}
	totalDim := m.Config.NumLayers * m.Config.HiddenPerLayer
	gotGGML := make([]float32, totalDim)
	if err := ggmlcompute.MulMatF16F32(gotGGML, wF16, hidden, m.Config.HiddenSize, totalDim); err != nil {
		t.Fatal(err)
	}
	gotGo := make([]float32, totalDim)
	gemvNT(gotGo, hidden, m.PerLayerModelProj, m.Config.HiddenSize, totalDim)
	hiddenRounded := make([]float32, len(hidden))
	for i, v := range hidden {
		hiddenRounded[i] = half.F16ToF32(half.F32ToF16(v))
	}
	gotGoRoundedInput := make([]float32, totalDim)
	gemvNT(gotGoRoundedInput, hiddenRounded, m.PerLayerModelProj, m.Config.HiddenSize, totalDim)

	maxGo, meanGo := maxMeanAbsDiff(gotGGML, gotGo)
	maxRounded, meanRounded := maxMeanAbsDiff(gotGGML, gotGoRoundedInput)
	t.Logf("real per_layer_model_proj ggml-vs-go max=%g mean=%g; ggml-vs-rounded-input max=%g mean=%g", maxGo, meanGo, maxRounded, meanRounded)
	if maxRounded > maxGo {
		t.Fatalf("rounded-input oracle is farther from ggml than current Go: rounded max=%g current max=%g", maxRounded, maxGo)
	}
	if maxRounded > 1e-4 {
		t.Fatalf("rounded-input oracle did not match ggml closely enough: max=%g mean=%g", maxRounded, meanRounded)
	}
	if maxGo < 1e-6 {
		t.Fatal("current Go unexpectedly matches ggml F16 mul_mat exactly; oracle no longer distinguishes input rounding")
	}
}

func flashAttnF16AccumReference(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, numHeads*headDim)
	headsPerKV := numHeads / numKVHeads
	kvDim := numKVHeads * headDim
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		qHead := q[head*headDim : (head+1)*headDim]
		qRounded := make([]float32, headDim)
		for i, v := range qHead {
			qRounded[i] = half.F16ToF32(half.F32ToF16(v))
		}
		S := float32(0)
		M := float32(math.Inf(-1))
		acc := make([]float32, headDim)
		for t := 0; t < seqLen; t++ {
			kHead := kCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			var score float32
			for d := 0; d < headDim; d++ {
				score += qRounded[d] * kHead[d]
			}
			score *= scale
			Mold := M
			vs := float32(1)
			if score > M {
				M = score
				ms := float32(math.Exp(float64(Mold - M)))
				for d := 0; d < headDim; d++ {
					acc[d] = half.F16ToF32(half.F32ToF16(acc[d] * ms))
				}
				S *= ms
			} else {
				vs = float32(math.Exp(float64(score - M)))
			}
			vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			for d := 0; d < headDim; d++ {
				acc[d] = half.F16ToF32(half.F32ToF16(acc[d] + vHead[d]*vs))
			}
			S += vs
		}
		invS := float32(0)
		if S != 0 {
			invS = 1 / S
		}
		for d := 0; d < headDim; d++ {
			out[head*headDim+d] = acc[d] * invS
		}
	}
	return out
}

func syntheticOracleVec(n int, scale float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i)*scale) * math.Cos(float64(i)*scale*0.37))
	}
	return out
}

func maxMeanAbsDiff(a, b []float32) (float64, float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var max, sum float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > max {
			max = d
		}
		sum += d
	}
	if n == 0 {
		return 0, 0
	}
	return max, sum / float64(n)
}
