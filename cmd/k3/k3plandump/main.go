package main

import (
	"flag"
	"fmt"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	path := flag.String("model", "", "GGUF model path")
	limit := flag.Int("limit", 20, "node print limit")
	flag.Parse()
	if *path == "" {
		panic("-model required")
	}
	m, err := model.LoadGGUFLlama(*path, k3.SIMDBackend{})
	if err != nil {
		panic(err)
	}
	g, p := m.DecodeGraph, m.DecodePlan
	fmt.Printf("graph: %s\n", g.Name)
	fmt.Printf("values: %d\n", len(g.Values))
	fmt.Printf("nodes: %d\n", len(g.Nodes))
	fmt.Printf("transient buffers: %d\n", len(p.Buffers))
	fmt.Printf("workspace: %.2f MiB\n", float64(p.WorkspaceBytes())/1024.0/1024.0)
	for i, n := range g.Nodes {
		if i >= *limit {
			fmt.Printf("... %d more nodes\n", len(g.Nodes)-i)
			break
		}
		fmt.Printf("%03d %-18s %-10s in=%v out=%v attrs=%v\n", i, n.Name, n.Op, n.Inputs, n.Outputs, n.Attrs)
	}
}
