package pockettts

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func tinyExportConfig(t *testing.T) Config {
	t.Helper()
	cfg := releasedTrainingConfig(t)
	cfg.FlowLM.Transformer.DModel = 4
	cfg.FlowLM.Transformer.NumHeads = 2
	cfg.FlowLM.Transformer.NumLayers = 1
	cfg.FlowLM.Transformer.HiddenScale = 2
	cfg.FlowLM.Transformer.DimFeedforward = 0
	cfg.FlowLM.Transformer.LayerScale = 0
	cfg.FlowLM.Flow.Dim = 4
	cfg.FlowLM.Flow.Depth = 2
	cfg.FlowLM.LookupTable.Dim = 4
	cfg.FlowLM.LookupTable.NBins = 5
	cfg.Mimi.InnerDim = 2
	cfg.Mimi.OuterDim = 2
	cfg.Mimi.SEANet.Dimension = 2
	cfg.Mimi.Transformer.DModel = 2
	cfg.Mimi.Transformer.NumHeads = 1
	cfg.Mimi.Transformer.NumLayers = 1
	cfg.Mimi.Transformer.DimFeedforward = 2
	cfg.Mimi.Transformer.InputDimension = 2
	cfg.Mimi.Transformer.OutputDimensions = []int{2}
	cfg.Mimi.Quantizer.Dimension = 2
	cfg.Mimi.Quantizer.OutputDimension = 2
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func tinyExportModels(t *testing.T) (*FullTrainer, Config) {
	t.Helper()
	cfg := tinyExportConfig(t)
	lm, flow := tinyFlowLMTraining(), tinyTrainableFlowHead()
	for i := range lm.Transformer.Layers {
		lm.Transformer.Layers[i].LayerScale1 = nil
		lm.Transformer.Layers[i].LayerScale2 = nil
		lm.Transformer.Layers[i].FC1 = tinyLinear(8, 4, .03)
		lm.Transformer.Layers[i].FC1.Bias = nil
		lm.Transformer.Layers[i].FC2 = tinyLinear(4, 8, -.015)
		lm.Transformer.Layers[i].FC2.Bias = nil
	}
	flow.Condition = tinyLinear(4, 4, -.02)
	for i := range flow.Time {
		flow.Time[i].Frequencies = make([]float32, 128)
		for j := range flow.Time[i].Frequencies {
			flow.Time[i].Frequencies[j] = float32(j+1) / 128
		}
		flow.Time[i].FC1 = tinyLinear(4, 256, .01)
	}
	weighting := &LSDWeightMLP{Layers: []LinearF32{{Weight: make([]float32, 64), Bias: make([]float32, 32), In: 2, Out: 32}, {Weight: make([]float32, 1024), Bias: make([]float32, 32), In: 32, Out: 32}, {Weight: make([]float32, 1024), Bias: make([]float32, 32), In: 32, Out: 32}, {Weight: make([]float32, 32), Bias: make([]float32, 1), In: 32, Out: 1}}}
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	return trainer, cfg
}

func writeFrozenMimiFixture(t *testing.T, cfg Config) string {
	t.Helper()
	required := expectedMimiStateShapes(cfg)
	tensors := make([]pocketExportTensor, 0, len(required))
	bf16Name := "mimi.quantizer.output_proj.weight"
	for name, shape := range required {
		n, ok := exportShapeElements(shape)
		if !ok {
			t.Fatal(name)
		}
		if name == bf16Name {
			raw := make([]byte, n*2)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint16(raw[i*2:], 0x3f80)
			}
			tensors = append(tensors, pocketExportTensor{name: name, dtype: "BF16", shape: shape, raw: raw})
		} else {
			values := make([]float32, n)
			for i := range values {
				values[i] = float32(i+1) / 100
			}
			tensors = append(tensors, pocketExportTensor{name: name, dtype: "F32", shape: shape, values: values})
		}
	}
	sortExportTensors(tensors)
	path := filepath.Join(t.TempDir(), "frozen.safetensors")
	if err := writePocketExport(path, tensors); err != nil {
		t.Fatal(err)
	}
	return path
}

func sortExportTensors(tensors []pocketExportTensor) {
	for i := 1; i < len(tensors); i++ {
		for j := i; j > 0 && tensors[j].name < tensors[j-1].name; j-- {
			tensors[j], tensors[j-1] = tensors[j-1], tensors[j]
		}
	}
}

func TestPinnedMimiStateShapeInventory(t *testing.T) {
	data, err := os.ReadFile("testdata/mimi_state_shapes_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema   int              `json:"schema"`
		Upstream string           `json:"upstream_revision"`
		Tensors  map[string][]int `json:"tensors"`
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Upstream != UpstreamCommit || len(fixture.Tensors) != 87 {
		t.Fatalf("fixture header=%+v tensors=%d", fixture, len(fixture.Tensors))
	}
	cfg := releasedTrainingConfig(t)
	got := expectedMimiStateShapes(cfg)
	if !reflect.DeepEqual(got, fixture.Tensors) {
		for name, want := range fixture.Tensors {
			if have, ok := got[name]; !ok || !equalShape(have, want) {
				t.Errorf("%s got=%v want=%v", name, have, want)
			}
		}
		for name, have := range got {
			if _, ok := fixture.Tensors[name]; !ok {
				t.Errorf("extra %s=%v", name, have)
			}
		}
		t.FailNow()
	}
}

func TestPocketSafetensorsExportDeterministicEMAAndFrozenMimi(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	frozenPath := writeFrozenMimiFixture(t, cfg)
	frozen, err := safetensors.Open(frozenPath)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	rawBefore, dtypeBefore, shapeBefore, err := frozen.GetRaw("mimi.quantizer.output_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	rawBefore = append([]byte(nil), rawBefore...)
	trainer.FlowLM.BOS[0] = 9
	trainer.EMA["flow_lm.bos_emb"][0] = 7
	trainer.FlowLM.LatentMean[0] = .75
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.safetensors"), filepath.Join(dir, "b.safetensors")
	if err = ExportPocketSafetensors(a, cfg, trainer, frozen, true); err != nil {
		t.Fatal(err)
	}
	if err = ExportPocketSafetensors(b, cfg, trainer, frozen, true); err != nil {
		t.Fatal(err)
	}
	ab, _ := os.ReadFile(a)
	bb, _ := os.ReadFile(b)
	if !reflect.DeepEqual(ab, bb) {
		t.Fatal("export is not deterministic")
	}
	out, err := safetensors.Open(a)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	for _, name := range out.Names() {
		if strings.HasPrefix(name, "flow.w_s_t") || strings.HasPrefix(name, "flow_lm.flow.w_s_t") {
			t.Fatalf("exported training-only tensor %s", name)
		}
	}
	bos, shape, err := out.GetFloat32("flow_lm.bos_emb")
	if err != nil || !equalShape(shape, []int{2}) || bos[0] != 7 {
		t.Fatalf("EMA BOS=%v shape=%v err=%v", bos, shape, err)
	}
	mean, _, _ := out.GetFloat32("flow_lm.emb_mean")
	if mean[0] != .75 {
		t.Fatalf("buffer did not stay live: %v", mean)
	}
	rawAfter, dtypeAfter, shapeAfter, err := out.GetRaw("mimi.quantizer.output_proj.weight")
	if err != nil || dtypeBefore != "BF16" || dtypeAfter != "F32" || !equalShape(shapeAfter, shapeBefore) || len(rawAfter) != 2*len(rawBefore) {
		t.Fatal("frozen Mimi tensor was not widened to upstream F32 state")
	}
	quantized, _, err := out.GetFloat32("mimi.quantizer.output_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range quantized {
		if v != 1 {
			t.Fatalf("widened Mimi value=%g", v)
		}
	}
	inv := InspectTensorInfos(cfg, out.TensorInfos())
	if !inv.CheckpointReady() {
		t.Fatalf("export inventory=%+v", inv)
	}
}

func TestPocketSafetensorsExportRawAndTransactionalFailure(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	frozen, err := safetensors.Open(writeFrozenMimiFixture(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	trainer.FlowLM.BOS[0] = 9
	trainer.EMA["flow_lm.bos_emb"][0] = 7
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err = ExportPocketSafetensors(path, cfg, trainer, frozen, false); err != nil {
		t.Fatal(err)
	}
	out, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bos, _, _ := out.GetFloat32("flow_lm.bos_emb")
	out.Close()
	if bos[0] != 9 {
		t.Fatalf("raw BOS=%v", bos)
	}
	sentinel := []byte("keep")
	if err = os.WriteFile(path, sentinel, 0o600); err != nil {
		t.Fatal(err)
	}
	delete(trainer.EMA, "flow_lm.bos_emb")
	if err = ExportPocketSafetensors(path, cfg, trainer, frozen, true); err != nil {
		t.Fatal(err)
	}
	partial, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bos, _, err = partial.GetFloat32("flow_lm.bos_emb")
	partial.Close()
	if err != nil || bos[0] != 9 {
		t.Fatalf("partial EMA did not fall back to raw: %v %v", bos, err)
	}
	if err = os.WriteFile(path, sentinel, 0o600); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	if !reflect.DeepEqual(got, sentinel) {
		t.Fatal("pre-validation changed sentinel")
	}
	trainer.EMADecay = 0
	trainer.EMA = nil
	if err = ExportPocketSafetensors(path, cfg, trainer, frozen, true); err == nil {
		t.Fatal("accepted missing EMA")
	}
	got, _ = os.ReadFile(path)
	if !reflect.DeepEqual(got, sentinel) {
		t.Fatal("failed export mutated destination")
	}
}

func TestPocketExportPublishedError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "published.safetensors")
	boom := errors.New("directory sync unavailable")
	calls := 0
	err := writePocketExportWithSync(path, []pocketExportTensor{{name: "x", dtype: "F32", shape: []int{1}, values: []float32{1.25}}}, func(*os.File) error {
		calls++
		if calls == 2 {
			return boom
		}
		return nil
	})
	var published *PocketExportPublishedError
	if !errors.As(err, &published) || published.Path != path || !errors.Is(err, boom) || calls != 2 {
		t.Fatalf("error=%#v calls=%d", err, calls)
	}
	file, openErr := safetensors.Open(path)
	if openErr != nil {
		t.Fatal(openErr)
	}
	values, shape, getErr := file.GetFloat32("x")
	file.Close()
	if getErr != nil || !equalShape(shape, []int{1}) || values[0] != 1.25 {
		t.Fatalf("published payload=%v shape=%v err=%v", values, shape, getErr)
	}
}

func TestPocketSafetensorsExportFlowMatchingOneTimeCondition(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	cfg.FlowLM.Flow.Type = "flow_matching"
	trainer.Flow.Time = trainer.Flow.Time[:1]
	trainer.topology.timeEmbeddings = 1
	params, err := fullParameterMap(trainer.FlowLM, trainer.Flow, trainer.Weighting)
	if err != nil {
		t.Fatal(err)
	}
	trainer.params = params
	trainer.names = sortedTrainingKeys(params)
	trainer.bindings = makeFullParamBindings(trainer.names, params)
	for name := range trainer.EMA {
		if strings.HasPrefix(name, "flow_lm.flow_net.time_embed.1.") {
			delete(trainer.EMA, name)
		}
	}
	frozen, err := safetensors.Open(writeFrozenMimiFixture(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	path := filepath.Join(t.TempDir(), "flow-matching.safetensors")
	if err = ExportPocketSafetensors(path, cfg, trainer, frozen, true); err != nil {
		t.Fatal(err)
	}
	out, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	for _, name := range out.Names() {
		if strings.HasPrefix(name, "flow_lm.flow_net.time_embed.1.") {
			t.Fatalf("exported second FlowMatching time condition %s", name)
		}
	}
}

func TestPocketSafetensorsExportRequiresExistingDirectory(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	frozen, err := safetensors.Open(writeFrozenMimiFixture(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	path := filepath.Join(t.TempDir(), "missing", "model.safetensors")
	if err = ExportPocketSafetensors(path, cfg, trainer, frozen, false); err == nil {
		t.Fatal("created unsynced parent directory")
	}
	if _, err = os.Stat(filepath.Dir(path)); !os.IsNotExist(err) {
		t.Fatalf("parent unexpectedly created: %v", err)
	}
}

func TestPocketSafetensorsExportRejectsIncompleteFrozenMimi(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	expected := expectedMimiStateShapes(cfg)
	tensors := make([]pocketExportTensor, 0, len(expected)-1)
	skipped := false
	for name, shape := range expected {
		if !skipped {
			skipped = true
			continue
		}
		n, _ := exportShapeElements(shape)
		tensors = append(tensors, pocketExportTensor{name: name, dtype: "F32", shape: shape, values: make([]float32, n)})
	}
	sortExportTensors(tensors)
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := writePocketExport(path, tensors); err != nil {
		t.Fatal(err)
	}
	frozen, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	if err = ExportPocketSafetensors(filepath.Join(t.TempDir(), "out.safetensors"), cfg, trainer, frozen, false); err == nil {
		t.Fatal("accepted incomplete frozen Mimi source")
	}
}

func TestPocketSafetensorsExportRejectsExtraFrozenMimi(t *testing.T) {
	trainer, cfg := tinyExportModels(t)
	source := writeFrozenMimiFixture(t, cfg)
	f, err := safetensors.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tensors := make([]pocketExportTensor, 0)
	for _, name := range f.Names() {
		raw, dtype, shape, e := f.GetRaw(name)
		if e != nil {
			t.Fatal(e)
		}
		tensors = append(tensors, pocketExportTensor{name: name, dtype: dtype, shape: append([]int(nil), shape...), raw: append([]byte(nil), raw...)})
	}
	tensors = append(tensors, pocketExportTensor{name: "mimi.extra", dtype: "F32", shape: []int{1}, values: []float32{1}})
	sortExportTensors(tensors)
	path := filepath.Join(t.TempDir(), "extra.safetensors")
	if err = writePocketExport(path, tensors); err != nil {
		t.Fatal(err)
	}
	extra, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer extra.Close()
	if err = ExportPocketSafetensors(filepath.Join(t.TempDir(), "out.safetensors"), cfg, trainer, extra, false); err == nil {
		t.Fatal("accepted extra frozen Mimi state")
	}
}
