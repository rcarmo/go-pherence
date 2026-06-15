package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/model"
)

type smokeResult struct {
	ModelDir                      string                     `json:"model_dir"`
	DrafterDir                    string                     `json:"drafter_dir"`
	ModelHidden                   int                        `json:"model_hidden"`
	ModelLayers                   int                        `json:"model_layers"`
	DrafterHidden                 int                        `json:"drafter_hidden"`
	DrafterBackbone               int                        `json:"drafter_backbone"`
	DrafterLayers                 int                        `json:"drafter_layers"`
	PackedEmbedding               bool                       `json:"packed_embedding"`
	PackedProjection              bool                       `json:"packed_projection"`
	PackedLayerWeights            bool                       `json:"packed_layer_weights"`
	Token                         int                        `json:"token"`
	LogitsLen                     int                        `json:"logits_len"`
	NextActivationLen             int                        `json:"next_activation_len"`
	LoadSeconds                   float64                    `json:"load_seconds"`
	StepSeconds                   float64                    `json:"step_seconds"`
	MTPGraphCapabilities          model.MTPGraphCapabilities `json:"mtp_graph_capabilities"`
	MTPMissingForPublicGeneration []string                   `json:"mtp_missing_for_public_generation,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "main Gemma4 model directory")
	drafterDir := flag.String("drafter", "", "Gemma4 MTP assistant/drafter directory")
	prevToken := flag.Int("token", 0, "previous token id for the drafter state")
	seqLen := flag.Int("seq", 1, "external KV sequence length")
	pretty := flag.Bool("pretty", true, "pretty-print JSON")
	flag.Parse()

	if *modelDir == "" || *drafterDir == "" {
		fmt.Fprintln(os.Stderr, "usage: gemma4mtpsmoke -model <main> -drafter <assistant>")
		os.Exit(2)
	}
	if *seqLen <= 0 {
		fmt.Fprintln(os.Stderr, "seq must be > 0")
		os.Exit(2)
	}

	model.ForceOnTheFly = true
	start := time.Now()
	m, err := model.LoadLlama(*modelDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	d, err := model.LoadGemma4MTPDrafter(*drafterDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load drafter: %v\n", err)
		os.Exit(1)
	}
	loadElapsed := time.Since(start)

	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		fmt.Fprintf(os.Stderr, "model/drafter mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d\n", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
		os.Exit(1)
	}

	k := make([][]float32, d.Config.NumLayers)
	v := make([][]float32, d.Config.NumLayers)
	for i := range d.Layers {
		headDim := d.Config.HeadDim
		if d.Layers[i].HeadDimLocal > 0 {
			headDim = d.Layers[i].HeadDimLocal
		}
		kvDim := d.Config.NumKVHeads * headDim
		k[i] = make([]float32, (*seqLen)*kvDim)
		v[i] = make([]float32, (*seqLen)*kvDim)
	}
	externalKV, err := model.NewMTPDrafterExternalKV(d, k, v, *seqLen)
	if err != nil {
		fmt.Fprintf(os.Stderr, "external KV: %v\n", err)
		os.Exit(1)
	}
	state, err := model.NewMTPDrafterState(*prevToken, make([]float32, d.BackboneHiddenSize), d.BackboneHiddenSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "state: %v\n", err)
		os.Exit(1)
	}

	stepStart := time.Now()
	step, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "drafter step: %v\n", err)
		os.Exit(1)
	}
	stepElapsed := time.Since(stepStart)

	packedLayer := len(d.Layers) > 0 && d.Layers[0].QWm != nil && d.Layers[0].OWm != nil && d.Layers[0].GateWm != nil && d.Layers[0].UpWm != nil && d.Layers[0].DownWm != nil
	caps := model.Gemma4MTPGraphCapabilities()
	res := smokeResult{
		ModelDir:                      *modelDir,
		DrafterDir:                    *drafterDir,
		ModelHidden:                   m.Config.HiddenSize,
		ModelLayers:                   len(m.Layers),
		DrafterHidden:                 d.Config.HiddenSize,
		DrafterBackbone:               d.BackboneHiddenSize,
		DrafterLayers:                 len(d.Layers),
		PackedEmbedding:               d.EmbedTokensMLX != nil,
		PackedProjection:              d.PreProjectionMLX != nil && d.PostProjectionMLX != nil,
		PackedLayerWeights:            packedLayer,
		Token:                         step.Token,
		LogitsLen:                     len(step.Logits),
		NextActivationLen:             len(step.NextActivation),
		LoadSeconds:                   loadElapsed.Seconds(),
		StepSeconds:                   stepElapsed.Seconds(),
		MTPGraphCapabilities:          caps,
		MTPMissingForPublicGeneration: caps.MissingForPublicGeneration(),
	}
	var out []byte
	if *pretty {
		out, _ = json.MarshalIndent(res, "", "  ")
	} else {
		out, _ = json.Marshal(res)
	}
	fmt.Println(string(out))
}
