package model

import (
	"fmt"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func makeSyntheticSelectedExpertBenchLayer(hidden, inter, numExperts int) *LlamaLayer {
	layer := &LlamaLayer{
		ExpertGateW: make([]*mlx.QuantWeight, numExperts),
		ExpertUpW:   make([]*mlx.QuantWeight, numExperts),
		ExpertDownW: make([]*mlx.QuantWeight, numExperts),
	}
	for i := 0; i < numExperts; i++ {
		layer.ExpertGateW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertUpW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertDownW[i] = benchMLXWeight(inter, hidden, 64)
	}
	return layer
}

func syntheticSelectedExpertWeights(n int) []float32 {
	out := make([]float32, n)
	if n <= 0 {
		return out
	}
	w := float32(1) / float32(n)
	for i := range out {
		out[i] = w
	}
	return out
}

func runSyntheticSelectedExpertCPUChain(out, gate, up, down, x []float32, layer *LlamaLayer, expertIDs []int, weights []float32) bool {
	clear(out)
	for i, expertID := range expertIDs {
		if !mlx.GemvTo(gate, x, layer.ExpertGateW[expertID]) || !mlx.GemvTo(up, x, layer.ExpertUpW[expertID]) {
			return false
		}
		simd.VecSiLUMul(gate, gate, up)
		if !mlx.GemvTo(down, gate, layer.ExpertDownW[expertID]) {
			return false
		}
		for j := range out {
			out[j] += weights[i] * down[j]
		}
	}
	return true
}

func runSyntheticSelectedExpertGPUChain(outBuf, xBuf, gateBuf, upBuf, downBuf *nvidia.DevBuf, entries []*nvidia.ExpertEntry, weights []float32) error {
	if outBuf == nil || xBuf == nil || gateBuf == nil || upBuf == nil || downBuf == nil {
		return fmt.Errorf("nil GPU selected-expert buffers")
	}
	if err := nvidia.ZeroFloat32Buffer(outBuf.GPUBuffer(), outBuf.Len()); err != nil {
		return err
	}
	initialized := false
	for i, entry := range entries {
		if entry == nil || entry.GateW == nil || entry.UpW == nil || entry.DownW == nil {
			return fmt.Errorf("nil GPU expert entry %d", i)
		}
		nvidia.GemvMLXDirect(gateBuf, xBuf, entry.GateW)
		nvidia.GemvMLXDirect(upBuf, xBuf, entry.UpW)
		nvidia.DevSiLUMul(gateBuf, gateBuf, upBuf)
		nvidia.GemvMLXDirect(downBuf, gateBuf, entry.DownW)
		if initialized {
			nvidia.DevAddScaled(outBuf, outBuf, downBuf, weights[i])
		} else {
			nvidia.DevScale(outBuf, downBuf, weights[i])
			initialized = true
		}
	}
	if !initialized {
		return fmt.Errorf("no selected GPU experts")
	}
	return nil
}

func maxAbsDiffSelectedExpertBench(got, want []float32) float32 {
	var max float32
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}

func BenchmarkMoESelectedExpertComputeSynthetic512x1024Top4(b *testing.B) {
	const (
		hidden     = 512
		inter      = 1024
		numExperts = 8
		active     = 4
		layerIdx   = 0
	)
	layer := makeSyntheticSelectedExpertBenchLayer(hidden, inter, numExperts)
	cfg := LlamaConfig{
		HiddenSize:       hidden,
		NumExperts:       numExperts,
		NumExpertsPerTok: active,
		MoEIntermediate:  inter,
		NormTopKProb:     true,
	}
	x := benchSeq(hidden)
	expertIDs := []int{0, 1, 2, 3}
	weights := syntheticSelectedExpertWeights(len(expertIDs))
	bytes := int64(len(expertIDs) * (2*hidden*inter + inter*hidden) * 4)

	if nvidia.SgemmReady() {
		xBuf := nvidia.NewDevBufFrom(append([]float32(nil), x...))
		if err := xBuf.ToGPU(); err != nil {
			b.Fatalf("upload x: %v", err)
		}
		defer xBuf.Free()

		pool := nvidia.NewExpertPool(len(expertIDs), nil)
		defer func() {
			for pool.Size() > 0 {
				nvidia.FreeExpertEntry(pool.EvictLRU())
			}
		}()

		entries := make([]*nvidia.ExpertEntry, len(expertIDs))
		for i, expertID := range expertIDs {
			poolKey := layerIdx*cfg.NumExperts + expertID
			entry := uploadExpertNativeToPool(pool, layer, expertID, poolKey, inter, hidden)
			if entry == nil {
				b.Fatalf("upload expert %d failed", expertID)
			}
			entries[i] = entry
		}

		generic := moeForwardGPU(nil, xBuf, layer, cfg, pool, layerIdx, nil)
		if len(generic) != hidden {
			b.Fatalf("generic GPU len=%d want %d", len(generic), hidden)
		}

		gateBuf, err := nvidia.NewDevBufGPU(inter)
		if err != nil {
			b.Fatal(err)
		}
		defer gateBuf.Free()
		upBuf, err := nvidia.NewDevBufGPU(inter)
		if err != nil {
			b.Fatal(err)
		}
		defer upBuf.Free()
		downBuf, err := nvidia.NewDevBufGPU(hidden)
		if err != nil {
			b.Fatal(err)
		}
		defer downBuf.Free()
		outBuf, err := nvidia.NewDevBufGPU(hidden)
		if err != nil {
			b.Fatal(err)
		}
		defer outBuf.Free()

		if err := runSyntheticSelectedExpertGPUChain(outBuf, xBuf, gateBuf, upBuf, downBuf, entries, weights); err != nil {
			b.Fatal(err)
		}
		if diff := maxAbsDiffSelectedExpertBench(outBuf.Data(), generic); diff > 1e-4 {
			b.Fatalf("GPU chain drift max_abs=%g", diff)
		}

		b.Run("generic_gpu_moe_warm_pool", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(bytes)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out := moeForwardGPU(nil, xBuf, layer, cfg, pool, layerIdx, nil)
				if len(out) != hidden {
					b.Fatalf("generic GPU len=%d", len(out))
				}
				nvidia.SyncForTiming()
			}
		})

		b.Run("per_expert_gpu_chain_warm_pool", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(bytes)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := runSyntheticSelectedExpertGPUChain(outBuf, xBuf, gateBuf, upBuf, downBuf, entries, weights); err != nil {
					b.Fatal(err)
				}
				nvidia.SyncForTiming()
			}
		})
		return
	}

	b.Run("generic_cpu_moe", func(b *testing.B) {
		want := runSyntheticSelectedExpertCPUChain(make([]float32, hidden), make([]float32, inter), make([]float32, inter), make([]float32, hidden), x, layer, expertIDs, weights)
		if !want {
			b.Fatal("CPU selected-expert chain preflight failed")
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := moeForward(x, layer, cfg)
			if len(out) != hidden {
				b.Fatalf("generic CPU len=%d", len(out))
			}
		}
	})

	b.Run("per_expert_cpu_chain", func(b *testing.B) {
		generic := moeForward(x, layer, cfg)
		if len(generic) != hidden {
			b.Fatalf("generic CPU len=%d want %d", len(generic), hidden)
		}
		out := make([]float32, hidden)
		gate := make([]float32, inter)
		up := make([]float32, inter)
		down := make([]float32, hidden)
		if !runSyntheticSelectedExpertCPUChain(out, gate, up, down, x, layer, expertIDs, weights) {
			b.Fatal("CPU selected-expert chain failed")
		}
		if diff := maxAbsDiffSelectedExpertBench(out, generic); diff != 0 {
			b.Fatalf("CPU chain drift max_abs=%g", diff)
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !runSyntheticSelectedExpertCPUChain(out, gate, up, down, x, layer, expertIDs, weights) {
				b.Fatal("CPU selected-expert chain failed")
			}
		}
	})
}
