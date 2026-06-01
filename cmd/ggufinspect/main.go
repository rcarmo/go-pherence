package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func main() {
	jsonOut := flag.Bool("json", false, "emit JSON")
	requirePure := flag.Bool("require-pure-go-simd-ready", false, "fail unless the GGUF has a pure Go/SIMD-compatible tensor index")
	requireRuntime := flag.Bool("require-runtime-ready", false, "fail unless current pure Go GGUF generation runtime has the expected tensors")
	requireMoE := flag.Bool("require-moe", false, "fail unless MoE metadata or tensors are present")
	requireQ4K := flag.Bool("require-q4-k", false, "fail unless Q4_K tensors are present")
	cacheTypeK := flag.String("cache-type-k", "", "validate native TurboQuant key cache type (turbo4, q8_0, f16)")
	cacheTypeV := flag.String("cache-type-v", "", "validate native TurboQuant value cache type (turbo2, q4_0, f16)")
	kvResidualWindow := flag.Int("kv-residual-window", -1, "native TurboQuant residual window for plan output")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: ggufinspect [flags] <model.gguf>")
		os.Exit(2)
	}
	in, err := gguf.Inspect(flag.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "ggufinspect: %v\n", err)
		os.Exit(1)
	}
	if *requirePure && !in.PureGoSIMDReady {
		fmt.Fprintln(os.Stderr, "ggufinspect: pure Go/SIMD readiness not satisfied")
		os.Exit(1)
	}
	if *requireRuntime && !in.RuntimeSupported {
		fmt.Fprintf(os.Stderr, "ggufinspect: runtime readiness not satisfied; missing tensors: %v\n", in.MissingRuntimeTensors)
		os.Exit(1)
	}
	if *requireMoE && !in.HasMoE {
		fmt.Fprintln(os.Stderr, "ggufinspect: MoE metadata/tensors not detected")
		os.Exit(1)
	}
	if *requireQ4K && !in.HasQ4K {
		fmt.Fprintln(os.Stderr, "ggufinspect: Q4_K tensors not detected")
		os.Exit(1)
	}
	var tqPlan any
	if *cacheTypeK != "" || *cacheTypeV != "" || *kvResidualWindow >= 0 {
		plan, err := ggufTurboQuantPlanFromInspection(in, *cacheTypeK, *cacheTypeV, *kvResidualWindow)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ggufinspect: turboquant plan: %v\n", err)
			os.Exit(1)
		}
		tqPlan = plan
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if tqPlan != nil {
			if err := enc.Encode(map[string]any{"inspection": in, "turboquant_plan": tqPlan}); err != nil {
				fmt.Fprintf(os.Stderr, "ggufinspect: encode JSON: %v\n", err)
				os.Exit(1)
			}
			return
		}
		if err := enc.Encode(in); err != nil {
			fmt.Fprintf(os.Stderr, "ggufinspect: encode JSON: %v\n", err)
			os.Exit(1)
		}
		return
	}
	fmt.Printf("path: %s\n", in.Path)
	fmt.Printf("architecture: %s\n", in.Architecture)
	if in.Name != "" {
		fmt.Printf("name: %s\n", in.Name)
	}
	fmt.Printf("tensors: %d\n", in.TensorCount)
	fmt.Printf("quant_counts: %v\n", in.QuantCounts)
	fmt.Printf("moe: %v experts=%d active=%d\n", in.HasMoE, in.Experts, in.ExpertsPerToken)
	fmt.Printf("reap_metadata: %v keys=%v\n", in.HasREAPMetadata, in.REAPMetadataKeys)
	fmt.Printf("turboquant_ready: %v\n", in.TurboQuantReady)
	fmt.Printf("pure_go_simd_ready: %v\n", in.PureGoSIMDReady)
	fmt.Printf("runtime_supported: %v\n", in.RuntimeSupported)
	if tqPlan != nil {
		fmt.Printf("turboquant_plan: %+v\n", tqPlan)
	}
	if len(in.MissingRuntimeTensors) > 0 {
		fmt.Printf("missing_runtime_tensors: %v\n", in.MissingRuntimeTensors)
	}
	for _, w := range in.ReadinessWarnings {
		fmt.Printf("warning: %s\n", w)
	}
}

type inspectTurboQuantPlan struct {
	Enabled          bool    `json:"enabled"`
	KeyType          string  `json:"key_type,omitempty"`
	ValueType        string  `json:"value_type,omitempty"`
	KeyBits          int     `json:"key_bits,omitempty"`
	ValueBits        int     `json:"value_bits,omitempty"`
	ResidualWindow   int     `json:"residual_window"`
	Layers           uint32  `json:"layers,omitempty"`
	CacheLayers      uint32  `json:"cache_layers,omitempty"`
	ProtectedLayers  int     `json:"protected_cache_layers,omitempty"`
	MaxSeqLen        uint32  `json:"max_seq_len,omitempty"`
	KVHeads          uint32  `json:"kv_heads,omitempty"`
	HeadDim          uint32  `json:"head_dim,omitempty"`
	KVDim            uint32  `json:"kv_dim,omitempty"`
	FullKVBytes      int64   `json:"full_kv_bytes,omitempty"`
	EstimatedBytes   int64   `json:"estimated_kv_bytes,omitempty"`
	SavedBytes       int64   `json:"estimated_saved_kv_bytes,omitempty"`
	EstimatedKVRatio float64 `json:"estimated_kv_ratio,omitempty"`
	RuntimeReady     bool    `json:"runtime_ready"`
}

func ggufTurboQuantPlanFromInspection(in gguf.Inspection, keyType, valueType string, residualWindow int) (inspectTurboQuantPlan, error) {
	cfg, enabled, err := kv.TurboQuantConfigFromCacheTypes(keyType, valueType, residualWindow)
	if err != nil {
		return inspectTurboQuantPlan{}, err
	}
	est := inspectTurboQuantEstimate(in, cfg, enabled)
	return inspectTurboQuantPlan{Enabled: enabled, KeyType: keyType, ValueType: valueType, KeyBits: cfg.KeyBits, ValueBits: cfg.ValueBits, ResidualWindow: cfg.ResidualWindow, Layers: in.Layers, CacheLayers: in.CompressedKVLayers, ProtectedLayers: est.ProtectedLayers, MaxSeqLen: in.MaxSeqLen, KVHeads: in.KVHeads, HeadDim: in.HeadDim, KVDim: in.KVDim, FullKVBytes: est.FullBytes, EstimatedBytes: est.EstimatedBytes, SavedBytes: est.SavedBytes, EstimatedKVRatio: est.Ratio, RuntimeReady: in.RuntimeSupported}, nil
}

func inspectTurboQuantEstimate(in gguf.Inspection, cfg kv.TurboQuantConfig, enabled bool) kv.TurboQuantKVEstimate {
	return kv.EstimateTurboQuantKV(int(in.Layers), int(in.KVHeads), int(in.HeadDim), int(in.MaxSeqLen), cfg, enabled, func(i int) bool { return inspectUsesKVLayer(in, uint32(i)) })
}

func inspectUsesKVLayer(in gguf.Inspection, layer uint32) bool {
	if layer >= in.Layers {
		return false
	}
	if in.FullAttentionInterval == 0 {
		return true
	}
	return (layer+1)%in.FullAttentionInterval == 0
}
