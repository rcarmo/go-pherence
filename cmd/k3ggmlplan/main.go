package main

import (
	"flag"
	"fmt"
	"sort"

	"github.com/rcarmo/go-pherence/backends/ggmlexec"
	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	minIsland := flag.Int("min-island", 2, "minimum nodes per reported GGML island")
	limit := flag.Int("limit", 40, "max islands/nodes to print")
	flag.Parse()
	if *path == "" {
		panic("-model required")
	}
	m, err := model.LoadGGUFLlama(*path, k3.SIMDBackend{})
	if err != nil {
		panic(err)
	}
	cov := ggmlexec.Analyze(m.DecodePlan, *minIsland)
	fmt.Printf("graph=%s\n", m.DecodeGraph.Name)
	fmt.Printf("nodes=%d values=%d buffers=%d workspace=%.2f MiB\n", len(m.DecodeGraph.Nodes), len(m.DecodeGraph.Values), len(m.DecodePlan.Buffers), float64(m.DecodePlan.WorkspaceBytes())/1024/1024)
	fmt.Printf("ggml-supported nodes=%d/%d (%.1f%%)\n", cov.SupportedNodes, cov.TotalNodes, 100*float64(cov.SupportedNodes)/float64(cov.TotalNodes))
	fmt.Println("unsupported ops:")
	var keys []string
	byName := map[string]int{}
	for k, v := range cov.Unsupported {
		keys = append(keys, string(k))
		byName[string(k)] = v
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("  %-12s %d\n", k, byName[k])
	}
	fmt.Printf("\nislands(min=%d)=%d\n", *minIsland, len(cov.Islands))
	for i, is := range cov.Islands {
		if i >= *limit {
			fmt.Printf("... %d more islands\n", len(cov.Islands)-i)
			break
		}
		fmt.Println(is.String())
		for j, n := range is.Nodes {
			if j >= *limit {
				fmt.Printf("    ... %d more nodes\n", len(is.Nodes)-j)
				break
			}
			fmt.Printf("    %03d %-24s %-10s\n", is.Start+j, n.Name, n.Op)
		}
	}
}
