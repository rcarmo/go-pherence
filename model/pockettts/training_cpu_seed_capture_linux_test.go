//go:build linux

package pockettts

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type cpuSeedCaptureBackend struct{ seed []float32 }

func (b *cpuSeedCaptureBackend) EvaluateTransformer(m *TransformerCPU, x, seed []float32, rows int) (TransformerTrainingResult, error) {
	out, g, dx, e := m.ForwardBackward(x, seed, rows)
	if e != nil {
		return TransformerTrainingResult{}, e
	}
	if !zeroTrainingSeed(seed) {
		b.seed = append([]float32(nil), seed...)
	}
	lm := &FlowLMTrainingGradients{Transformer: g}
	all := map[string][]float32{}
	addFlowLMGradients(all, lm)
	p := map[string][]float32{}
	for n, v := range all {
		if strings.HasPrefix(n, "flow_lm.transformer.") || strings.HasPrefix(n, "flow_lm.out_norm.") {
			p[n] = v
		}
	}
	return TransformerTrainingResult{out, dx, p}, nil
}
func TestPocketCPUSeedCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_CPU_SEED_CAPTURE") != "1" {
		t.Skip("independent CPU fullstep seed arithmetic")
	}
	in := allrowPrepare(t)
	_, _, w, _ := releasedTrainingModels(t)
	b := &cpuSeedCaptureBackend{}
	in.LM.TransformerBackend = b
	// Independent per-row arithmetic at the exact CPU conditioner values.
	lmForward, err := in.LM.forwardTrainingTape(in.Batch, make([]float32, in.Batch.Frames*in.LM.Hidden), make([]float32, in.Batch.Frames))
	if err != nil {
		t.Fatal(err)
	}
	diag, dist, _ := productionObjectiveScales(in.Valid)
	diagRows, distRows := make([]float32, 14*1024), make([]float32, 14*1024)
	var forwards []parameterRow
	for row := 0; row < in.Valid; row++ {
		c := lmForward.output.Z[row*1024 : (row+1)*1024]
		target, noise := in.Batch.NormalizedLatents[row*32:(row+1)*32], in.Samples.Noise[row*32:(row+1)*32]
		xs, desired := make([]float32, 32), make([]float32, 32)
		tm := in.Samples.DiagonalTime[row]
		for i := range xs {
			xs[i] = tm*target[i] + (1-tm)*noise[i]
			desired[i] = target[i] - noise[i]
		}
		tape, pred, e := in.Flow.forwardTraining(c, []float32{tm, tm}, xs)
		if e != nil {
			t.Fatal(e)
		}
		_, seed, _, e := LSDDiagonalLossAndGradient(pred, desired, []float32{0}, 32, true)
		if e != nil {
			t.Fatal(e)
		}
		for i := range seed {
			seed[i] *= diag
		}
		dc, _, _, e := in.Flow.backwardTraining(tape, seed, newFlowHeadGradients(in.Flow))
		if e != nil {
			t.Fatal(e)
		}
		copy(diagRows[row*1024:], dc)
		_, d, e := in.Flow.lsdDistillForwardBackwardInto(c, in.Samples.DistillS[row], in.Samples.DistillT[row], noise, target, 0, newFlowHeadGradients(in.Flow), newFlowHeadGradients(in.Flow), dist)
		if e != nil {
			t.Fatal(e)
		}
		copy(distRows[row*1024:], d.Condition)
		s, dt := in.Samples.DistillS[row], in.Samples.DistillT[row]
		px := make([]float32, 32)
		for i := range px {
			px[i] = s*target[i] + (1-s)*noise[i]
		}
		pt := in.Flow.forwardDualTape(c, []float32{s, dt}, px, 1)
		epx := make([]float32, 32)
		for i := range epx {
			epx[i] = px[i] + (dt-s)*pt.final.output.value[i]
		}
		_, ep, e := in.Flow.forwardTraining(c, []float32{dt, dt}, epx)
		if e != nil {
			t.Fatal(e)
		}
		forwards = append(forwards, parameterRow{row, xs, pred, pt.final.output.value, pt.final.output.tangent, epx, ep})
	}
	if err = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_CPU_SEED_OUTPUT"), "cpu-row-seeds.json"), struct{ Diagonal, Distill []float32 }{diagRows, distRows}); err != nil {
		t.Fatal(err)
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_CPU_SEED_OUTPUT"), "cpu-row-forward.json"), forwards); e != nil {
		t.Fatal(e)
	}
	_, g, e := PocketTrainingStep(in.LM, in.Flow, w, in.Batch, in.Samples, DefaultTrainingStepConfig())
	if e != nil {
		t.Fatal(e)
	}
	params := map[string][]float32{}
	addFlowLMGradients(params, g.FlowLM)
	addFlowHeadGradientParameters(params, g.Flow)
	for i, l := range g.Weighting.Layers {
		prefix := fmt.Sprintf("flow.w_s_t.%d", 2*i)
		params[prefix+".weight"] = l.Weight
		params[prefix+".bias"] = l.Bias
	}
	var sum, carry float64
	var count int64
	for _, v := range params {
		for _, x := range v {
			term := float64(x)*float64(x) - carry
			next := sum + term
			carry = (next - sum) - term
			sum = next
			count++
		}
	}
	scale := float32(1 / math.Sqrt(sum))
	root := os.Getenv("GO_PHERENCE_POCKET_CPU_SEED_OUTPUT")
	if e = flowCaptureWrite(filepath.Join(root, "cpu-seed.json"), struct {
		TrunkSeed []float32
		Norm      float64
		Scale     float32
	}{b.seed, math.Sqrt(sum), scale}); e != nil {
		t.Fatal(e)
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	initial, e := fullParameterMap(lm, flow, w)
	if e != nil {
		t.Fatal(e)
	}
	path := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c", 1431217112)
	r, e := sharedOpenCheckpointP(path)
	if e != nil {
		t.Fatal(e)
	}
	defer r.file.Close()
	var max float64
	over := 0
	var raw [32768]byte
	for index := 0; index < r.count; index++ {
		name, n, e := r.next()
		if e != nil || len(params[name]) != n || len(initial[name]) != n {
			t.Fatal("inventory", name, e)
		}
		for off := 0; off < n; {
			c := min(n-off, len(raw)/4)
			if _, e = io.ReadFull(r.file, raw[:4*c]); e != nil {
				t.Fatal(e)
			}
			for j := 0; j < c; j++ {
				want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
				p, e := parameterFirstStepPrediction(initial[name][off+j], params[name][off+j], scale, DefaultAdamWConfig())
				if e != nil {
					t.Fatal(e)
				}
				d := math.Abs(float64(p) - float64(want))
				max = math.Max(max, d)
				if d > 4e-6 {
					over++
				}
			}
			off += c
		}
	}
	t.Logf("independentCPUvalues=%d max=%g over4e-6=%d seed=%d", count, max, over, len(b.seed))
	if over != 0 {
		t.Fatal("CPU oracle reproduction", over, max)
	}
}
