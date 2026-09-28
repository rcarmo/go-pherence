package pockettts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

type depthDistillOracle struct {
	Schema             int                  `json:"schema"`
	Upstream           string               `json:"upstream_revision"`
	Frames             int                  `json:"frames"`
	VoiceFrames        int                  `json:"voice_frames"`
	Hidden             int                  `json:"hidden"`
	LatentDim          int                  `json:"latent_dim"`
	Vocabulary         int                  `json:"vocabulary"`
	CFGCoef            float32              `json:"cfg_coef"`
	Mask               []bool               `json:"mask"`
	ShiftedMask        []bool               `json:"shifted_mask"`
	NormalizedLatents  []float32            `json:"normalized_latents"`
	VoiceLatents       []float32            `json:"voice_latents"`
	TextTokens         []uint32             `json:"text_tokens"`
	StudentLayers      int                  `json:"student_layers"`
	TeacherLayers      int                  `json:"teacher_layers"`
	Selected           []int                `json:"selected_24_to_6"`
	Loss               float64              `json:"loss"`
	Student            []float32            `json:"student"`
	TeacherConditioned []float32            `json:"teacher_conditioned"`
	TeacherNull        []float32            `json:"teacher_null"`
	Target             []float32            `json:"target"`
	DNormalizedLatents []float32            `json:"d_normalized_latents"`
	DVoiceLatents      []float32            `json:"d_voice_latents"`
	StudentParameters  map[string][]float32 `json:"student_parameters"`
	TeacherParameters  map[string][]float32 `json:"teacher_parameters"`
	StudentGradients   map[string][]float32 `json:"student_gradients"`
}

func loadDepthDistillOracle(t testing.TB) depthDistillOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/depth_distill_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var o depthDistillOracle
	if err = json.Unmarshal(data, &o); err != nil {
		t.Fatal(err)
	}
	if o.Schema != 1 || o.Upstream != UpstreamCommit || o.Frames != 3 || o.StudentLayers != 1 || o.TeacherLayers != 2 {
		t.Fatal("unexpected depth distill oracle")
	}
	return o
}

func flowLMFromDepthOracle(t testing.TB, parameters map[string][]float32, layers, hidden, latent, vocab int) *FlowLMTrainingCPU {
	t.Helper()
	linear := func(prefix string, out, in int, bias bool) LinearF32 {
		w := parameters[prefix+".weight"]
		if len(w) != out*in {
			t.Fatalf("%s=%d", prefix, len(w))
		}
		l := LinearF32{Weight: append([]float32(nil), w...), In: in, Out: out}
		if bias {
			l.Bias = append([]float32(nil), parameters[prefix+".bias"]...)
		}
		return l
	}
	model := &TransformerCPU{Width: hidden, Heads: 2, HeadDim: hidden / 2, Context: 0, MaxPeriod: 100, Layers: make([]TransformerLayerCPU, layers), FinalWeight: append([]float32(nil), parameters["out_norm.weight"]...), FinalBias: append([]float32(nil), parameters["out_norm.bias"]...)}
	for i := range model.Layers {
		p := fmtLayer(i)
		model.Layers[i] = TransformerLayerCPU{Norm1Weight: append([]float32(nil), parameters[p+".norm1.weight"]...), Norm1Bias: append([]float32(nil), parameters[p+".norm1.bias"]...), Norm2Weight: append([]float32(nil), parameters[p+".norm2.weight"]...), Norm2Bias: append([]float32(nil), parameters[p+".norm2.bias"]...), InProjection: linear(p+".self_attn.in_proj", 3*hidden, hidden, false), OutProjection: linear(p+".self_attn.out_proj", hidden, hidden, false), FC1: linear(p+".linear1", 6, hidden, false), FC2: linear(p+".linear2", hidden, 6, false), LayerScale1: append([]float32(nil), parameters[p+".layer_scale_1.scale"]...), LayerScale2: append([]float32(nil), parameters[p+".layer_scale_2.scale"]...)}
	}
	return &FlowLMTrainingCPU{Embedding: append([]float32(nil), parameters["conditioner.embed.weight"]...), Vocabulary: vocab, Hidden: hidden, LatentDim: latent, BOS: append([]float32(nil), parameters["bos_emb"]...), BOSBeforeVoice: append([]float32(nil), parameters["bos_before_voice"]...), SpeakerProjection: LinearF32{Weight: append([]float32(nil), parameters["speaker_proj_weight"]...), In: latent, Out: hidden}, Input: linear("input_linear", hidden, latent, false), Transformer: model, EOS: linear("out_eos", 1, hidden, true)}
}
func fmtLayer(i int) string { return fmt.Sprintf("transformer.layers.%d", i) }

func TestDepthDistillPyTorchParity(t *testing.T) {
	o := loadDepthDistillOracle(t)
	student := flowLMFromDepthOracle(t, o.StudentParameters, o.StudentLayers, o.Hidden, o.LatentDim, o.Vocabulary)
	teacher := flowLMFromDepthOracle(t, o.TeacherParameters, o.TeacherLayers, o.Hidden, o.LatentDim, o.Vocabulary)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	result, gradients, err := DepthDistillForwardBackward(student, teacher, batch, o.Mask, o.CFGCoef)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result.Loss-o.Loss) > 2e-7 {
		t.Fatalf("loss=%g want=%g", result.Loss, o.Loss)
	}
	if !reflect.DeepEqual(result.ShiftedMask, o.ShiftedMask) {
		t.Fatalf("shifted=%v want=%v", result.ShiftedMask, o.ShiftedMask)
	}
	assertSliceClose(t, "student", result.Student, o.Student, 7e-6)
	assertSliceClose(t, "teacher conditioned", result.TeacherConditioned, o.TeacherConditioned, 7e-6)
	assertSliceClose(t, "teacher null", result.TeacherNull, o.TeacherNull, 7e-6)
	assertSliceClose(t, "teacher target", result.Target, o.Target, 8e-6)
	assertSliceClose(t, "distill audio gradient", gradients.Inputs.NormalizedLatents, o.DNormalizedLatents, 2e-5)
	assertSliceClose(t, "distill voice gradient", gradients.Inputs.VoiceLatents, o.DVoiceLatents, 2e-5)
	got := flowLMTrainingGradientMap(gradients.Student)
	for _, v := range append(append([]float32(nil), gradients.Student.EOS.Weight...), gradients.Student.EOS.Bias...) {
		if v != 0 {
			t.Fatalf("frozen EOS gradient=%g", v)
		}
	}
	delete(got, "out_eos.weight")
	delete(got, "out_eos.bias")
	if len(got) != len(o.StudentGradients) {
		t.Fatalf("gradients=%d want=%d", len(got), len(o.StudentGradients))
	}
	for name, value := range got {
		want, ok := o.StudentGradients[name]
		if !ok {
			t.Fatalf("oracle lacks %s", name)
		}
		assertSliceClose(t, "depth distill "+name, value, want, 3e-5)
	}
}

func TestSelectDistillationLayersAndShrink(t *testing.T) {
	got, err := SelectDistillationLayers(24, 6)
	if err != nil || !reflect.DeepEqual(got, []int{0, 1, 2, 21, 22, 23}) {
		t.Fatalf("got=%v err=%v", got, err)
	}
	state := map[string][]float32{"flow_lm.bos_emb": {9}, "flow_lm.transformer.layers.0.x": {0}, "flow_lm.transformer.layers.1.x": {1}, "flow_lm.transformer.layers.2.x": {2}, "flow_lm.transformer.layers.3.x": {3}, "flow_lm.transformer.layers.4.x": {4}, "flow_lm.transformer.layers.5.x": {5}, "flow_lm.transformer.layers.6.x": {6}, "flow_lm.transformer.layers.7.x": {7}}
	shrunk, kept, err := ShrinkFlowLMState(state, 3)
	if err != nil || !reflect.DeepEqual(kept, []int{0, 1, 7}) {
		t.Fatalf("kept=%v err=%v", kept, err)
	}
	for name, want := range map[string]float32{"flow_lm.transformer.layers.0.x": 0, "flow_lm.transformer.layers.1.x": 1, "flow_lm.transformer.layers.2.x": 7, "flow_lm.bos_emb": 9} {
		if got := shrunk[name]; len(got) != 1 || got[0] != want {
			t.Fatalf("%s=%v", name, got)
		}
	}
	shrunk["flow_lm.bos_emb"][0] = 0
	if state["flow_lm.bos_emb"][0] != 9 {
		t.Fatal("shrink aliased source")
	}
}

func TestDistillTrainerLeavesFrozenHeadsUnchanged(t *testing.T) {
	o := loadDepthDistillOracle(t)
	student := flowLMFromDepthOracle(t, o.StudentParameters, 1, o.Hidden, o.LatentDim, o.Vocabulary)
	teacher := flowLMFromDepthOracle(t, o.TeacherParameters, 2, o.Hidden, o.LatentDim, o.Vocabulary)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	_, gradients, err := DepthDistillForwardBackward(student, teacher, batch, o.Mask, o.CFGCoef)
	if err != nil {
		t.Fatal(err)
	}
	frozenEOS := append(append([]float32(nil), student.EOS.Weight...), student.EOS.Bias...)
	flow := tinyTrainableFlowHead()
	weighting := &LSDWeightMLP{Layers: []LinearF32{{Weight: []float32{.2, .3}, Bias: []float32{.1}, In: 2, Out: 1}}}
	frozenFlow, _ := fullParameterMap(student, flow, weighting)
	snap := cloneTrainingMap(frozenFlow)
	trainableBefore := cloneTrainingMap(distillParameterMap(student))
	trainer, err := NewDistillTrainer(student, AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}, .9)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients.Student); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(distillParameterMap(student), trainableBefore) {
		t.Fatal("student trainable subset did not update")
	}
	if !reflect.DeepEqual(append(append([]float32(nil), student.EOS.Weight...), student.EOS.Bias...), frozenEOS) {
		t.Fatal("EOS moved during distillation")
	}
	after, _ := fullParameterMap(student, flow, weighting)
	for name, before := range snap {
		if strings.HasPrefix(name, "flow_lm.flow_net.") || strings.HasPrefix(name, "flow.w_s_t.") {
			if !reflect.DeepEqual(after[name], before) {
				t.Fatalf("frozen %s moved", name)
			}
		}
	}
	if len(trainer.EMA) != len(distillParameterMap(student)) {
		t.Fatalf("EMA tracks frozen params: %d", len(trainer.EMA))
	}
}

func TestDistillTrainerCopiesAliasedGradients(t *testing.T) {
	o := loadDepthDistillOracle(t)
	makeCase := func() (*FlowLMTrainingCPU, *FlowLMTrainingGradients) {
		student := flowLMFromDepthOracle(t, o.StudentParameters, 1, o.Hidden, o.LatentDim, o.Vocabulary)
		teacher := flowLMFromDepthOracle(t, o.TeacherParameters, 2, o.Hidden, o.LatentDim, o.Vocabulary)
		batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
		_, g, err := DepthDistillForwardBackward(student, teacher, batch, o.Mask, o.CFGCoef)
		if err != nil {
			t.Fatal(err)
		}
		return student, g.Student
	}
	aliased, gAlias := makeCase()
	plain, gPlain := makeCase()
	copy(gPlain.BOS, aliased.BOS)
	gAlias.BOS = aliased.BOS
	config := AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}
	ta, _ := NewDistillTrainer(aliased, config, .9)
	tp, _ := NewDistillTrainer(plain, config, .9)
	if err := ta.Step(gAlias); err != nil {
		t.Fatal(err)
	}
	if err := tp.Step(gPlain); err != nil {
		t.Fatal(err)
	}
	for name, want := range distillParameterMap(plain) {
		if !reflect.DeepEqual(distillParameterMap(aliased)[name], want) {
			t.Fatalf("aliased parameter %s differs", name)
		}
	}
	if !reflect.DeepEqual(ta.Moments, tp.Moments) || !reflect.DeepEqual(ta.EMA, tp.EMA) {
		t.Fatal("aliased optimizer state differs")
	}
}

func TestDistillTrainerRebindSafety(t *testing.T) {
	o := loadDepthDistillOracle(t)
	student := flowLMFromDepthOracle(t, o.StudentParameters, 1, o.Hidden, o.LatentDim, o.Vocabulary)
	teacher := flowLMFromDepthOracle(t, o.TeacherParameters, 2, o.Hidden, o.LatentDim, o.Vocabulary)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	_, gradients, err := DepthDistillForwardBackward(student, teacher, batch, o.Mask, o.CFGCoef)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewDistillTrainer(student, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	old := student.BOS
	student.BOS = append([]float32(nil), old...)
	before := student.BOS[0]
	if err = trainer.Step(gradients.Student); err != nil {
		t.Fatal(err)
	}
	if student.BOS[0] == before || old[0] != before {
		t.Fatal("same-model slice rebind did not follow live parameter")
	}
	replacement := flowLMFromDepthOracle(t, o.StudentParameters, 1, o.Hidden, o.LatentDim, o.Vocabulary)
	trainer.Student = replacement
	if err = trainer.Step(gradients.Student); err == nil {
		t.Fatal("accepted student pointer replacement")
	}
	trainer.Student = student
	student.Transformer.Layers = append(student.Transformer.Layers, student.Transformer.Layers[0])
	if err = trainer.Step(gradients.Student); err == nil {
		t.Fatal("accepted expanded student topology")
	}
	trainer.Student = nil
	if err = trainer.Step(gradients.Student); err == nil {
		t.Fatal("accepted nil student")
	}
}

func TestShrinkFlowLMStateRealistic24To6(t *testing.T) {
	state := map[string][]float32{"transformer.layers.23.norm1.weight": {23}, "transformer.layers.21.norm1.weight": {21}, "transformer.layers.10.norm1.weight": {10}, "transformer.layers.2.norm1.weight": {2}, "transformer.layers.1.norm1.weight": {1}, "transformer.layers.0.norm1.weight": {0}, "bos_emb": {99}}
	shrunk, kept, err := ShrinkFlowLMState(state, 6)
	if err != nil || !reflect.DeepEqual(kept, []int{0, 1, 2, 21, 22, 23}) {
		t.Fatalf("kept=%v err=%v", kept, err)
	}
	for name, want := range map[string]float32{"transformer.layers.0.norm1.weight": 0, "transformer.layers.1.norm1.weight": 1, "transformer.layers.2.norm1.weight": 2, "transformer.layers.3.norm1.weight": 21, "transformer.layers.5.norm1.weight": 23, "bos_emb": 99} {
		if got := shrunk[name]; len(got) != 1 || got[0] != want {
			t.Fatalf("%s=%v", name, got)
		}
	}
	if _, ok := shrunk["transformer.layers.10.norm1.weight"]; ok {
		t.Fatal("retained middle teacher layer")
	}
}

func TestDepthDistillRejectsMalformedWithoutMutation(t *testing.T) {
	o := loadDepthDistillOracle(t)
	student := flowLMFromDepthOracle(t, o.StudentParameters, 1, o.Hidden, o.LatentDim, o.Vocabulary)
	teacher := flowLMFromDepthOracle(t, o.TeacherParameters, 2, o.Hidden, o.LatentDim, o.Vocabulary)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: append([]float32(nil), o.NormalizedLatents...), VoiceLatents: append([]float32(nil), o.VoiceLatents...), TextTokens: o.TextTokens}
	original := append([]float32(nil), batch.NormalizedLatents...)
	cases := []struct {
		mask []bool
		cfg  float32
	}{{nil, 2}, {[]bool{false, false, false}, 2}, {o.Mask, 0}, {o.Mask, float32(math.NaN())}}
	for i, c := range cases {
		if _, _, err := DepthDistillForwardBackward(student, teacher, batch, c.mask, c.cfg); err == nil {
			t.Fatalf("case %d accepted", i)
		}
		if !reflect.DeepEqual(batch.NormalizedLatents, original) {
			t.Fatalf("case %d mutated batch", i)
		}
	}
	for _, pair := range [][2]int{{0, 1}, {1, 2}, {2, 3}} {
		if _, err := SelectDistillationLayers(pair[0], pair[1]); err == nil {
			t.Fatalf("accepted %v", pair)
		}
	}
	if _, _, err := ShrinkFlowLMState(map[string][]float32{"x": {1}}, 1); err == nil {
		t.Fatal("accepted state without layers")
	}
}
