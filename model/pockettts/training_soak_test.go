package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"
)

type trainingSoakSample struct {
	Run         string `json:"run"`
	Step        int    `json:"step"`
	HeapAlloc   uint64 `json:"heap_alloc"`
	HeapInuse   uint64 `json:"heap_inuse"`
	HeapObjects uint64 `json:"heap_objects"`
}
type trainingSoakReport struct {
	Schema                int                  `json:"schema"`
	Commit                string               `json:"commit"`
	Fixture               string               `json:"fixture"`
	StepsPerRun           int                  `json:"steps_per_run"`
	ExecutedUpdates       int                  `json:"executed_updates"`
	CheckpointInterval    int                  `json:"checkpoint_interval"`
	ElapsedSeconds        float64              `json:"elapsed_seconds"`
	FinalStateSHA256      string               `json:"final_state_sha256"`
	FinalCheckpointSHA256 string               `json:"final_checkpoint_sha256"`
	ForwardBackwardAllocs float64              `json:"forward_backward_allocs"`
	OptimizerAllocs       float64              `json:"optimizer_allocs"`
	Samples               []trainingSoakSample `json:"samples"`
}

func TestPocketTrainingDeterministicTinySoak(t *testing.T) {
	stepsText := os.Getenv("GO_PHERENCE_POCKETTTS_SOAK_STEPS")
	if stepsText == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_SOAK_STEPS for the deterministic tiny-graph soak")
	}
	steps, err := strconv.Atoi(stepsText)
	if err != nil || steps < 1000 {
		t.Fatalf("invalid soak steps %q", stepsText)
	}
	interval := steps / 10
	if interval < 1 {
		interval = 1
	}
	oracle := loadTrainingStepOracle(t)
	newRun := func() (*FullTrainer, *TrainingWorkspace, *FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, FlowLMTrainingBatch, TrainingStepSamples) {
		lm, flow, w := trainingStepModelsFromOracle(t, oracle)
		batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: append([]float32(nil), oracle.NormalizedLatents...), VoiceLatents: append([]float32(nil), oracle.VoiceLatents...), TextTokens: append([]uint32(nil), oracle.TextTokens...)}
		samples := TrainingStepSamples{Mask: append([]bool(nil), oracle.Mask...), Noise: append([]float32(nil), oracle.Noise...), DiagonalTime: append([]float32(nil), oracle.DiagonalTime...), DistillS: append([]float32(nil), oracle.DistillS...), DistillT: append([]float32(nil), oracle.DistillT...)}
		workspace, e := NewTrainingWorkspace(lm, flow, batch)
		if e != nil {
			t.Fatal(e)
		}
		trainer, e := NewFullTrainer(lm, flow, w, AdamWConfig{LearningRate: oracle.AdamW.LearningRate, Beta1: oracle.AdamW.Beta1, Beta2: oracle.AdamW.Beta2, Epsilon: oracle.AdamW.Epsilon, WeightDecay: oracle.AdamW.WeightDecay}, oracle.AdamW.EMADecay)
		if e != nil {
			t.Fatal(e)
		}
		return trainer, workspace, lm, flow, w, batch, samples
	}
	vary := func(samples TrainingStepSamples, step int) {
		phase := float32(step%97) / 97
		for i, base := range oracle.Noise {
			samples.Noise[i] = base + (phase-.5)*.02
		}
		for row := range samples.DiagonalTime {
			samples.DiagonalTime[row] = .1 + .8*float32((step+row*17)%89)/89
			s := .05 + .45*float32((step+row*11)%83)/83
			tValue := s + .5*float32((step+row*7)%79)/79
			if tValue > 1 {
				tValue = 1
			}
			samples.DistillS[row], samples.DistillT[row] = s, tValue
		}
	}
	stepRun := func(step int, trainer *FullTrainer, workspace *TrainingWorkspace, lm *FlowLMTrainingCPU, flow *FlowHeadCPU, w *LSDWeightMLP, batch FlowLMTrainingBatch, samples TrainingStepSamples) {
		vary(samples, step)
		_, g, e := PocketTrainingStepInto(lm, flow, w, batch, samples, DefaultTrainingStepConfig(), workspace)
		if e != nil {
			t.Fatal(e)
		}
		if e = trainer.Step(g); e != nil {
			t.Fatal(e)
		}
	}
	sampleHeap := func(run string, step int) trainingSoakSample {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return trainingSoakSample{Run: run, Step: step, HeapAlloc: m.HeapAlloc, HeapInuse: m.HeapInuse, HeapObjects: m.HeapObjects}
	}
	stateBytes := func(trainer *FullTrainer) []byte {
		state, e := trainer.State()
		if e != nil {
			t.Fatal(e)
		}
		data, e := json.Marshal(state)
		if e != nil {
			t.Fatal(e)
		}
		return data
	}
	// Recheck established warm allocation ceilings in this executable.
	at, aw, alm, af, alw, ab, as := newRun()
	vary(as, 1)
	var warm TrainingStepGradients
	forwardAllocs := testing.AllocsPerRun(20, func() {
		_, warm, err = PocketTrainingStepInto(alm, af, alw, ab, as, DefaultTrainingStepConfig(), aw)
		if err != nil {
			panic(err)
		}
	})
	if forwardAllocs > 720 {
		t.Fatalf("soak forward allocations=%g", forwardAllocs)
	}
	if err = at.Step(warm); err != nil {
		t.Fatal(err)
	}
	optimizerAllocs := testing.AllocsPerRun(100, func() {
		if e := at.Step(warm); e != nil {
			panic(e)
		}
	})
	if optimizerAllocs != 0 {
		t.Fatalf("soak optimizer allocations=%g", optimizerAllocs)
	}
	commit := "unknown"
	if raw, e := exec.Command("git", "rev-parse", "HEAD").Output(); e == nil {
		commit = string(raw)
		for len(commit) > 0 && (commit[len(commit)-1] == '\n' || commit[len(commit)-1] == '\r') {
			commit = commit[:len(commit)-1]
		}
	}
	report := trainingSoakReport{Schema: 2, Commit: commit, Fixture: "model/pockettts/testdata/training_step_pytorch.json", StepsPerRun: steps, ExecutedUpdates: 2 * steps, CheckpointInterval: interval, ForwardBackwardAllocs: forwardAllocs, OptimizerAllocs: optimizerAllocs}
	started := time.Now()
	continuous, cw, clm, cf, clw, batch, samples := newRun()
	report.Samples = append(report.Samples, sampleHeap("continuous", 0))
	boundary := map[int][]byte{}
	for step := 1; step <= steps; step++ {
		stepRun(step, continuous, cw, clm, cf, clw, batch, samples)
		if step%interval == 0 || step == steps {
			boundary[step] = stateBytes(continuous)
			report.Samples = append(report.Samples, sampleHeap("continuous", step))
		}
	}
	resumed, rw, rlm, rf, rlw, batch, samples := newRun()
	report.Samples = append(report.Samples, sampleHeap("resumed", 0))
	checkpointDir := t.TempDir()
	checkpoint := filepath.Join(checkpointDir, "state.json")
	for step := 1; step <= steps; step++ {
		stepRun(step, resumed, rw, rlm, rf, rlw, batch, samples)
		if step%interval == 0 || step == steps {
			before := stateBytes(resumed)
			if string(before) != string(boundary[step]) {
				t.Fatalf("resume path diverged before checkpoint at step %d", step)
			}
			state, e := resumed.State()
			if e != nil {
				t.Fatal(e)
			}
			if e = SaveFullTrainingState(checkpoint, state); e != nil {
				t.Fatal(e)
			}
			loaded, e := LoadFullTrainingState(checkpoint)
			if e != nil {
				t.Fatal(e)
			}
			rlm, rf, rlw = trainingStepModelsFromOracle(t, oracle)
			resumed, e = NewFullTrainer(rlm, rf, rlw, state.AdamW, state.EMADecay)
			if e != nil {
				t.Fatal(e)
			}
			if e = resumed.LoadState(loaded); e != nil {
				t.Fatal(e)
			}
			rw, e = NewTrainingWorkspace(rlm, rf, batch)
			if e != nil {
				t.Fatal(e)
			}
			after := stateBytes(resumed)
			if string(after) != string(boundary[step]) {
				t.Fatalf("loaded checkpoint diverged at step %d", step)
			}
			report.Samples = append(report.Samples, sampleHeap("resumed", step))
		}
	}
	final := boundary[steps]
	stateSum := sha256.Sum256(final)
	report.FinalStateSHA256 = hex.EncodeToString(stateSum[:])
	checkpointBytes, err := os.ReadFile(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	checkpointSum := sha256.Sum256(checkpointBytes)
	report.FinalCheckpointSHA256 = hex.EncodeToString(checkpointSum[:])
	report.ElapsedSeconds = time.Since(started).Seconds()
	for _, run := range []string{"continuous", "resumed"} {
		var baseline *trainingSoakSample
		for i := range report.Samples {
			if report.Samples[i].Run == run && report.Samples[i].Step == 0 {
				baseline = &report.Samples[i]
				break
			}
		}
		if baseline == nil {
			t.Fatal("missing step-zero heap sample")
		}
		const byteAllowance = 8 << 20
		const objectAllowance = 10000
		for _, sample := range report.Samples {
			if sample.Run != run {
				continue
			}
			if sample.HeapInuse > baseline.HeapInuse+byteAllowance || sample.HeapAlloc > baseline.HeapAlloc+byteAllowance || sample.HeapObjects > baseline.HeapObjects+objectAllowance {
				t.Fatalf("%s retained heap sample exceeds allowance: baseline=%+v sample=%+v", run, *baseline, sample)
			}
		}
	}
	if out := os.Getenv("GO_PHERENCE_POCKETTTS_SOAK_REPORT"); out != "" {
		data, e := json.MarshalIndent(report, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(out, append(data, '\n'), 0o600); e != nil {
			t.Fatal(e)
		}
	}
	t.Logf("Pocket TTS tiny soak steps/run=%d executed=%d elapsed=%s state=%s checkpoint=%s", steps, 2*steps, time.Since(started), report.FinalStateSHA256, report.FinalCheckpointSHA256)
}
