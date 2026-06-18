//go:build ggml && cgo && linux

package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGemma4Layer0ActualPLIBranchGGMLOracle(t *testing.T) {
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
	m, ctx, batch, qkv, kvK, kvV := gemma4LayerQKVAndKVForFlashOracle(t, 0)
	row := 1
	seqLen := ctx.SeqLen + row + 1
	kCache := append([]float32(nil), kvK[0]...)
	vCache := append([]float32(nil), kvV[0]...)
	for i := 0; i <= row; i++ {
		kCache = append(kCache, qkv.K[i*qkv.KVDim:(i+1)*qkv.KVDim]...)
		vCache = append(vCache, qkv.V[i*qkv.KVDim:(i+1)*qkv.KVDim]...)
	}
	_, _, kRounded, vRounded := ggmlF16KVFromGoCache(kCache, vCache, seqLen, qkv.KVHeads, qkv.HeadDim)
	layer := m.Layers[0]
	attn := ggmlFlashAttnF16KVReference(qkv.Q[row*qkv.QDim:(row+1)*qkv.QDim], kRounded, vRounded, seqLen, m.Config.NumHeads, qkv.KVHeads, qkv.HeadDim, 1.0)
	o := make([]float32, m.Config.HiddenSize)
	if !gemvGGUFTo(o, attn, layer.OWGGUF, qkv.QDim, m.Config.HiddenSize) {
		t.Fatal("O projection rejected")
	}
	rmsNormInPlace(o, layer.PostNorm.Data(), float32(m.Config.RMSNormEps))
	peIn := append([]float32(nil), batch.HiddenRows[row]...)
	simd.VecAdd(peIn, peIn, o)
	ffnNorm := append([]float32(nil), peIn...)
	rmsNormInPlace(ffnNorm, layer.PreFFNNorm.Data(), float32(m.Config.RMSNormEps))
	gate := make([]float32, layer.GateWGGUF.OutDim)
	up := make([]float32, layer.UpWGGUF.OutDim)
	if !gemvGGUFTo(gate, ffnNorm, layer.GateWGGUF, m.Config.HiddenSize, len(gate)) || !gemvGGUFTo(up, ffnNorm, layer.UpWGGUF, m.Config.HiddenSize, len(up)) {
		t.Fatal("gate/up rejected")
	}
	ggmlGELUMulInPlace(gate, up)
	down := make([]float32, m.Config.HiddenSize)
	if !gemvGGUFTo(down, gate, layer.DownWGGUF, len(gate), m.Config.HiddenSize) {
		t.Fatal("down rejected")
	}
	rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(m.Config.RMSNormEps))
	simd.VecAdd(peIn, peIn, down)

	pli := batch.PerLayerInputs[row][0]
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	loadRaw := func(name string) ([]byte, int) {
		t.Helper()
		tensor, ok := g.TensorByName(name)
		if !ok {
			t.Fatalf("%s not found", name)
		}
		raw, err := g.Raw(tensor)
		if err != nil {
			t.Fatal(err)
		}
		return raw, int(tensor.QType)
	}
	rawGate, qtGate := loadRaw("blk.0.inp_gate.weight")
	gateGo := make([]float32, m.Config.HiddenPerLayer)
	if !gemvGGUFTo(gateGo, peIn, layer.PLIGateGGUF, m.Config.HiddenSize, m.Config.HiddenPerLayer) {
		t.Fatal("PLI gate rejected")
	}
	gateGGML := make([]float32, m.Config.HiddenPerLayer)
	if err := ggmlcompute.MulMatQuantF32(qtGate, gateGGML, rawGate, peIn, layer.PLIGateGGUF.InDim, layer.PLIGateGGUF.OutDim); err != nil {
		t.Fatal(err)
	}
	if maxDiff, meanDiff := maxMeanAbsDiff(gateGGML, gateGo); maxDiff > 1e-4 {
		t.Fatalf("PLI gate actual max=%g mean=%g", maxDiff, meanDiff)
	}
	actGGML := make([]float32, len(gateGo))
	if err := ggmlcompute.GEGLUSplitF32(actGGML, gateGo, pli); err != nil {
		t.Fatal(err)
	}
	ggmlGELUMulInPlace(gateGo, pli)
	if maxDiff, meanDiff := maxMeanAbsDiff(actGGML, gateGo); maxDiff > 3e-4 {
		t.Fatalf("PLI gelu*input actual max=%g mean=%g", maxDiff, meanDiff)
	}
	rawProj, qtProj := loadRaw("blk.0.proj.weight")
	projGo := make([]float32, m.Config.HiddenSize)
	if !gemvGGUFTo(projGo, gateGo, layer.PLIProjGGUF, m.Config.HiddenPerLayer, m.Config.HiddenSize) {
		t.Fatal("PLI proj rejected")
	}
	projGGML := make([]float32, m.Config.HiddenSize)
	if err := ggmlcompute.MulMatQuantF32(qtProj, projGGML, rawProj, gateGo, layer.PLIProjGGUF.InDim, layer.PLIProjGGUF.OutDim); err != nil {
		t.Fatal(err)
	}
	if maxDiff, meanDiff := maxMeanAbsDiff(projGGML, projGo); maxDiff > 1e-4 {
		t.Fatalf("PLI proj actual max=%g mean=%g", maxDiff, meanDiff)
	}
	postGGML := make([]float32, len(projGo))
	if err := ggmlcompute.RMSNormMulF32(postGGML, projGo, layer.PLIPostNorm, float32(m.Config.RMSNormEps)); err != nil {
		t.Fatal(err)
	}
	rmsNormInPlace(projGo, layer.PLIPostNorm, float32(m.Config.RMSNormEps))
	if maxDiff, meanDiff := maxMeanAbsDiff(postGGML, projGo); maxDiff > 1e-5 {
		t.Fatalf("PLI post norm actual max=%g mean=%g", maxDiff, meanDiff)
	}
}
