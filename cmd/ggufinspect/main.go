package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	jsonOut := flag.Bool("json", false, "emit JSON")
	requirePure := flag.Bool("require-pure-go-simd-ready", false, "fail unless the GGUF has a pure Go/SIMD-compatible tensor index")
	requireMoE := flag.Bool("require-moe", false, "fail unless MoE metadata or tensors are present")
	requireQ4K := flag.Bool("require-q4-k", false, "fail unless Q4_K tensors are present")
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
	if *requireMoE && !in.HasMoE {
		fmt.Fprintln(os.Stderr, "ggufinspect: MoE metadata/tensors not detected")
		os.Exit(1)
	}
	if *requireQ4K && !in.HasQ4K {
		fmt.Fprintln(os.Stderr, "ggufinspect: Q4_K tensors not detected")
		os.Exit(1)
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
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
	for _, w := range in.ReadinessWarnings {
		fmt.Printf("warning: %s\n", w)
	}
}
