package qwenimage21

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func fixtureConfig() Config {
	return Config{ModelIndex: ModelIndex{ClassName: "QwenImage21Pipeline", Processor: []string{"transformers", "Qwen3VLProcessor"}, Scheduler: []string{"diffusers", "FlowMatchEulerDiscreteScheduler"}, TextEncoder: []string{"transformers", "Qwen3VLForConditionalGeneration"}, Transformer: []string{"diffusers", "QwenImage21Transformer2DModel"}, VAE: []string{"diffusers", "AutoencoderKLQwenImage21"}}, Transformer: TransformerConfig{ClassName: "QwenImage21Transformer2DModel", AttentionHeadDim: 128, AxesDimsRoPE: []int{16, 56, 56}, ContextInDim: 4096, InChannels: 64, NumAttentionHeads: 32, NumLayers: 32, OutChannels: 64, PatchSize: 1, MLPRatio: 3, Eps: 1e-6, CausalCondition: true}, Scheduler: SchedulerConfig{ClassName: "FlowMatchEulerDiscreteScheduler", BaseImageSeqLen: 256, BaseShift: .5, MaxImageSeqLen: 8192, MaxShift: .9, NumTrainTimesteps: 1000, Shift: 1, ShiftTerminal: .02, TimeShiftType: "exponential", UseDynamicShifting: true}, VAE: VAEConfig{ClassName: "AutoencoderKLQwenImage21", InChannels: 4, OutChannels: 4, BaseDim: 96, DecoderBaseDim: 144, DimMult: []int{1, 2, 4, 8, 8}, NumResBlocks: 2, ZDim: 64, ScaleFactorSpatial: 16, ScaleFactorTemporal: 8, LatentsMean: make([]float64, 64), LatentsStd: func() []float64 {
		x := make([]float64, 64)
		for i := range x {
			x[i] = 1
		}
		return x
	}(), TemporalDownsample: []bool{false, true, true, true}}}
}
func TestConfigAndRead(t *testing.T) {
	c := fixtureConfig()
	if e := ValidateConfig(c); e != nil {
		t.Fatal(e)
	}
	if c.HiddenSize() != 4096 || c.IntermediateSize() != 12288 {
		t.Fatal(c.HiddenSize(), c.IntermediateSize())
	}
	h, w, e := c.LatentGrid(1024, 768)
	if e != nil || h != 64 || w != 48 {
		t.Fatal(h, w, e)
	}
	if _, _, e = c.LatentGrid(31, 32); e == nil {
		t.Fatal("dims")
	}
	root := t.TempDir()
	write := func(name, stringValue string) {
		t.Helper()
		p := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(p), 0700)
		os.WriteFile(p, []byte(stringValue), 0600)
	}
	write("model_index.json", `{"_class_name":"QwenImage21Pipeline","processor":["transformers","Qwen3VLProcessor"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"text_encoder":["transformers","Qwen3VLForConditionalGeneration"],"transformer":["diffusers","QwenImage21Transformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage21"]}`)
	write("transformer/config.json", `{"_class_name":"QwenImage21Transformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],"context_in_dim":4096,"in_channels":64,"num_attention_heads":32,"num_layers":32,"out_channels":64,"patch_size":1,"mlp_ratio":3,"eps":0.000001,"causal_condition":true}`)
	write("scheduler/scheduler_config.json", `{"_class_name":"FlowMatchEulerDiscreteScheduler","base_image_seq_len":256,"base_shift":0.5,"max_image_seq_len":8192,"max_shift":0.9,"num_train_timesteps":1000,"shift":1,"shift_terminal":0.02,"time_shift_type":"exponential","use_dynamic_shifting":true}`)
	write("vae/config.json", `{"_class_name":"AutoencoderKLQwenImage21","in_channels":4,"out_channels":4,"base_dim":96,"decoder_base_dim":144,"dim_mult":[1,2,4,8,8],"num_res_blocks":2,"z_dim":64,"scale_factor_spatial":16,"scale_factor_temporal":8,"latents_mean":[`+repeat("0", 64)+`],"latents_std":[`+repeat("1", 64)+`]}`)
	got, e := ReadConfig(root)
	if e != nil || got.Transformer.NumLayers != 32 {
		t.Fatal(got, e)
	}
}
func repeat(v string, n int) string {
	s := ""
	for i := 0; i < n; i++ {
		if i > 0 {
			s += ","
		}
		s += v
	}
	return s
}
func TestConfigRejects(t *testing.T) {
	base := fixtureConfig()
	cases := []func(*Config){func(c *Config) { c.ModelIndex.ClassName = "x" }, func(c *Config) { c.Transformer.InChannels = 16 }, func(c *Config) { c.Scheduler.UseDynamicShifting = false }, func(c *Config) { c.VAE.ZDim = 16 }, func(c *Config) { c.VAE.LatentsStd[2] = 0 }}
	for _, mut := range cases {
		c := base
		c.Transformer.AxesDimsRoPE = append([]int(nil), base.Transformer.AxesDimsRoPE...)
		c.VAE.LatentsMean = append([]float64(nil), base.VAE.LatentsMean...)
		c.VAE.LatentsStd = append([]float64(nil), base.VAE.LatentsStd...)
		mut(&c)
		if e := ValidateConfig(c); e == nil {
			t.Fatal("accepted bad config")
		}
	}
}
func TestLayoutTextOnly(t *testing.T) {
	l, e := BuildLayout(3, nil, []ImageShape{{2, 2}})
	if e != nil {
		t.Fatal(e)
	}
	if l.PrefixLength != 3 || len(l.Positions) != 7 || len(l.Segments) != 2 {
		t.Fatal(l)
	}
	want := []Position{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, -1, -1}, {3, -1, 0}, {3, 0, -1}, {3, 0, 0}}
	if !slices.Equal(l.Positions, want) {
		t.Fatal(l.Positions)
	}
	ids, target := l.TokenMetadata()
	if !slices.Equal(ids, []int{-1, -1, -1, 0, 0, 0, 0}) || !slices.Equal(target, []bool{false, false, false, true, true, true, true}) {
		t.Fatal(ids, target)
	}
	if !l.AttentionAllowed(3, 6) || l.AttentionAllowed(1, 2) || !l.AttentionAllowed(6, 1) {
		t.Fatal("mask")
	}
}
func TestLayoutReferences(t *testing.T) {
	l, e := BuildLayout(4, []int{0, 1, 0, 0}, []ImageShape{{2, 2}, {1, 2}})
	if e != nil {
		t.Fatal(e)
	}
	if l.PrefixLength != 7 || len(l.Positions) != 9 || len(l.Segments) != 4 {
		t.Fatal(l)
	}
	ids, target := l.TokenMetadata()
	if ids[1] != 0 || ids[4] != 0 || ids[7] != 1 || !target[7] || target[4] {
		t.Fatal(ids, target)
	}
	for _, tc := range []struct {
		n     int
		slots []int
		sh    []ImageShape
	}{{1, []int{1}, []ImageShape{{2, 2}}}, {2, []int{1}, []ImageShape{{1, 1}, {1, 1}}}, {2, []int{2, 2}, []ImageShape{{1, 1}, {1, 1}}}} {
		if _, e := BuildLayout(tc.n, tc.slots, tc.sh); e == nil {
			t.Fatal("bad layout")
		}
	}
}
func TestScheduler(t *testing.T) {
	c := fixtureConfig().Scheduler
	mu, e := CalculateShift(256, c)
	if e != nil || math.Abs(mu-.5) > 1e-12 {
		t.Fatal(mu, e)
	}
	steps, e := Schedule(256, 4, c)
	if e != nil || len(steps) != 4 || steps[0].Sigma != 1 || steps[3].SigmaNext != 0 || math.Abs(float64(steps[3].Sigma-.02)) > 1e-6 {
		t.Fatal(steps, e)
	}
	dst := make([]float32, 2)
	if e := EulerStep(dst, []float32{1, 2}, []float32{3, 4}, FlowStep{Delta: -.5}); e != nil || !slices.Equal(dst, []float32{-.5, 0}) {
		t.Fatal(dst, e)
	}
	if e := CFG(dst, []float32{3, 5}, []float32{1, 1}, 2); e != nil || !slices.Equal(dst, []float32{5, 9}) {
		t.Fatal(dst, e)
	}
	if _, e := Schedule(1, 0, c); e == nil {
		t.Fatal("steps")
	}
	if _, e := CalculateShift(0, c); e == nil {
		t.Fatal("shift")
	}
	if e := EulerStep(nil, nil, nil, FlowStep{}); e == nil {
		t.Fatal("shape")
	}
	if e := CFG(nil, nil, nil, 1); e == nil {
		t.Fatal("cfg shape")
	}
	if e := CFG(make([]float32, 1), []float32{1}, []float32{1}, float32(math.NaN())); e == nil {
		t.Fatal("cfg nan")
	}
}
func TestOps(t *testing.T) {
	dst := make([]float32, 2)
	if e := ZeroCenteredRMSNorm(dst, []float32{3, 4}, []float32{0, 0}, 1e-6); e != nil {
		t.Fatal(e)
	}
	if math.Abs(float64(dst[0]-.848528)) > 1e-5 {
		t.Fatal(dst)
	}
	if e := LayerNorm(dst, []float32{1, 3}, 1e-6); e != nil || math.Abs(float64(dst[0]+1)) > 1e-5 {
		t.Fatal(dst, e)
	}
	time, e := TimestepEmbedding(0, 4)
	if e != nil || !slices.Equal(time, []float32{1, 1, 0, 0}) {
		t.Fatal(time, e)
	}
	x := []float32{1, 2, 3, 4}
	norm := []float32{0, 0}
	iw := []float32{1, 0, 0, 1}
	ow := []float32{1, 0, 0, 1}
	out := make([]float32, 4)
	if e := TextProjection(out, x, norm, iw, ow, 2, 2); e != nil {
		t.Fatal(e)
	}
	mod := make([]float32, 4)
	if e := Modulate(mod, x, []float32{0, 0, 1, 1}, 1, false); e != nil || !slices.Equal(mod, []float32{1, 2, 6, 8}) {
		t.Fatal(mod, e)
	}
	if e := Modulate(mod, x, []float32{0, 0, 1, 1}, 1, true); e != nil {
		t.Fatal(e)
	}
	if e := Modulate(nil, x, nil, 0, false); e == nil {
		t.Fatal("mod shape")
	}
	if _, e := TimestepEmbedding(0, 3); e == nil {
		t.Fatal("time dim")
	}
	if e := ZeroCenteredRMSNorm(nil, nil, nil, 1); e == nil {
		t.Fatal("rms shape")
	}
	if e := LayerNorm(nil, nil, 1); e == nil {
		t.Fatal("ln shape")
	}
	if e := TextProjection(nil, x, norm, iw, ow, 2, 2); e == nil {
		t.Fatal("projection shape")
	}
}
func TestRunnerSuccess(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "x")
	os.WriteFile(f, []byte("x"), 0600)
	script := filepath.Join(dir, "run")
	os.WriteFile(script, []byte("#!/bin/sh\nout=''\nwhile [ $# -gt 0 ];do if [ \"$1\" = '-o' ];then out=$2;shift 2;else shift;fi;done\nprintf '\\211PNG\\r\\n\\032\\nfixture' > \"$out\"\n"), 0700)
	r := Runner{Executable: script, Assets: Assets{DiffusionModel: f, VAE: f, LLM: f}}
	if got := tail("abcdef", 3); got != "def" {
		t.Fatal(got)
	}
	out := filepath.Join(dir, "o.png")
	got, e := r.Generate(context.Background(), GenerateOptions{Prompt: "x", Output: out, Width: 32, Height: 32, Steps: 1, Guidance: 1})
	if e != nil || got.Bytes != 15 || got.SHA256 == "" {
		t.Fatal(got, e)
	}
}
func TestRunnerAdmission(t *testing.T) {
	f := filepath.Join(t.TempDir(), "x")
	os.WriteFile(f, []byte("x"), 0700)
	r := Runner{Executable: f, Assets: Assets{DiffusionModel: f, VAE: f, LLM: f}}
	if e := r.Validate(); e != nil {
		t.Fatal(e)
	}
	for _, o := range []GenerateOptions{{}, {Prompt: "x", Output: "x", Width: 31, Height: 32, Steps: 1}, {Prompt: "x", Output: "x", Width: 32, Height: 32, Steps: 0}, {Prompt: "x", Output: "x", Width: 32, Height: 32, Steps: 1, Guidance: -1}} {
		if _, e := r.Generate(context.Background(), o); e == nil {
			t.Fatal("accepted options")
		}
	}
	fail := filepath.Join(t.TempDir(), "fail")
	os.WriteFile(fail, []byte("#!/bin/sh\necho failure >&2\nexit 1\n"), 0700)
	r.Executable = fail
	if _, e := r.Generate(context.Background(), GenerateOptions{Prompt: "x", Output: filepath.Join(t.TempDir(), "o.png"), Width: 32, Height: 32, Steps: 1, Timeout: time.Second}); e == nil {
		t.Fatal("backend failure")
	}
}
func TestRealInventoryOptIn(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_QWEN_IMAGE21_DIFFUSION")
	if path == "" {
		t.Skip("set Qwen Image 2.1 diffusion weights")
	}
	var inv Inventory
	var e error
	if filepath.Ext(path) == ".gguf" {
		inv, e = InspectGGUF(path)
	} else {
		inv, e = InspectSafetensors(path)
	}
	if e != nil || !inv.Ready || inv.Layers != 32 {
		t.Fatal(inv, e)
	}
}
func expectedShapes() map[string][]int {
	return map[string][]int{"img_in.weight": {64, 4096}, "modulation.1.weight": {4096, 16384}, "norm_out.linear.weight": {4096, 4096}, "proj_out.weight": {4096, 64}, "time_text_embed.timestep_embedder.linear_1.weight": {256, 4096}, "time_text_embed.timestep_embedder.linear_2.weight": {4096, 4096}, "txt_in.in_layer.weight": {4096, 4096}, "txt_in.out_layer.weight": {4096, 4096}, "txt_in.text_norm.weight": {4096}, "attn.norm_k.weight": {128}, "attn.norm_q.weight": {128}, "attn.to_k.weight": {4096, 4096}, "attn.to_out.0.weight": {4096, 4096}, "attn.to_q.weight": {4096, 4096}, "attn.to_v.weight": {4096, 4096}, "img_mlp.gate_up.weight": {4096, 24576}, "img_mlp.out.weight": {12288, 4096}}
}
func TestInspectSafetensorsRejectsMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.safetensors")
	if e := os.WriteFile(path, []byte("bad"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := InspectSafetensors(path); e == nil {
		t.Fatal("malformed safetensors")
	}
}

func TestInventory(t *testing.T) {
	var items []TensorInfo
	sh := expectedShapes()
	for _, n := range globalTensors {
		items = append(items, TensorInfo{Name: n, DType: "BF16", Shape: sh[n]})
	}
	for i := 0; i < 32; i++ {
		for _, s := range layerSuffixes {
			items = append(items, TensorInfo{Name: "transformer_blocks." + fmt.Sprint(i) + "." + s, DType: "Q2_K", Shape: sh[s]})
		}
	}
	inv := Inspect(items, "fixture")
	if !inv.Ready || inv.Layers != 32 || inv.Quantized != 256 {
		t.Fatal(inv)
	}
	inv = Inspect(items[:len(items)-1], "fixture")
	if inv.Ready || len(inv.Missing) != 1 {
		t.Fatal(inv)
	}
}
