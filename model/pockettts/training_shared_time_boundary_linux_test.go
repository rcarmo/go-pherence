//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type sharedTimeBoundaryRow struct {
	Branch                                                                string `json:"branch"`
	Row, Embedding, FrequencyIndex                                        int
	Time, TimeTangent, Frequency, Angle, Cos, Sin, CosTangent, SinTangent float32
}

// Only input preparation and fixed-host formula boundaries are evaluated.
// No flow, transformer, optimizer or device arithmetic is executed here.
func TestPocketSharedReleasedTimeBoundaryInputs(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_TIME_BOUNDARY") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_TIME_BOUNDARY=1 with pinned model/corpus/CPU run and new evidence directory")
	}
	pinned := func(path, hash string, limit int64) []byte {
		t.Helper()
		info, err := os.Stat(path)
		if err != nil || info.Size() > limit {
			t.Fatal("boundary input size", err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		if hex.EncodeToString(sum[:]) != hash {
			t.Fatal("boundary input pin", path)
		}
		return raw
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(pinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20), &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal("corpus", err)
	}
	var saved struct {
		Identity TrainingRunIdentity `json:"identity"`
		TimeRNG  string              `json:"time_rng"`
		NoiseRNG string              `json:"noise_rng"`
	}
	if err := json.Unmarshal(pinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16), &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Identity.Backend != "simd" || saved.Identity.GoVersion != runtime.Version() || saved.Identity.GOARCH != runtime.GOARCH || saved.Identity.GOOS != runtime.GOOS {
		t.Fatal("sampling platform")
	}
	row := corpus.Rows[0]
	if row.Batch.Frames != 14 || row.ValidFrames != 13 || row.Batch.VoiceFrames != 13 || len(row.Batch.TextTokens) != 1 {
		t.Fatal("fixture shape")
	}
	run, err := NewTrainingRunState(saved.Identity, 1, [6]uint64{42, 43, 44, 45, 46, 47})
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := run.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	samples := runnerSampleBuffers(row.Batch.Frames, 32)
	recipe := TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: saved.Identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
	if err := SampleTrainingInto(samples, proposal, row.Batch.Frames, row.ValidFrames, 32, recipe, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}); err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(proposal.next.timeRNG[:]) != saved.TimeRNG || hex.EncodeToString(proposal.next.noiseRNG[:]) != saved.NoiseRNG {
		t.Fatal("sampling boundary mismatch")
	}
	modelPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	src, err := safetensors.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	frequencies := make([][]float32, 2)
	for i, name := range []string{"flow_lm.flow_net.time_embed.0.freqs", "flow_lm.flow_net.time_embed.1.freqs"} {
		v, shape, err := src.GetFloat32(name)
		if err != nil || !equalShape(shape, []int{128}) || !finiteF32(v) {
			t.Fatal("frequency input", name, err)
		}
		frequencies[i] = v
	}
	report := struct {
		Scope                                                      string
		Rows                                                       []sharedTimeBoundaryRow
		MaxAngle, MaxValueReferenceError, MaxTangentReferenceError float64
		RopeValues, RopeChanged                                    int
		RopeMaxError                                               float64
	}{Scope: "Pinned released angles and host formula boundaries. No device sin/cos/sqrt/exp execution or native equivalence."}
	for r := 0; r < row.ValidFrames; r++ {
		for _, branch := range []struct {
			name  string
			times [2]float32
			index int
		}{{"diagonal", [2]float32{samples.DiagonalTime[r], samples.DiagonalTime[r]}, -1}, {"primary", [2]float32{samples.DistillS[r], samples.DistillT[r]}, 1}, {"endpoint", [2]float32{samples.DistillT[r], samples.DistillT[r]}, -1}} {
			for emb := 0; emb < 2; emb++ {
				dt := float32(0)
				if emb == branch.index {
					dt = 1
				}
				for j, f := range frequencies[emb] {
					angle := branch.times[emb] * f
					report.MaxAngle = max(report.MaxAngle, math.Abs(float64(angle)))
					if !isFinite(angle) || math.Abs(float64(angle)) > 8192 {
						t.Fatal("actual device-time admission")
					}
					sin, cos := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
					ct, st := (-sin*f)*dt, (cos*f)*dt
					report.MaxValueReferenceError = max(report.MaxValueReferenceError, math.Abs(float64(sin)-math.Sin(float64(angle))), math.Abs(float64(cos)-math.Cos(float64(angle))))
					report.MaxTangentReferenceError = max(report.MaxTangentReferenceError, math.Abs(float64(ct)-(-math.Sin(float64(angle))*float64(f)*float64(dt))), math.Abs(float64(st)-(math.Cos(float64(angle))*float64(f)*float64(dt))))
					report.Rows = append(report.Rows, sharedTimeBoundaryRow{Branch: branch.name, Row: r, Embedding: emb, FrequencyIndex: j, Time: branch.times[emb], TimeTangent: dt, Frequency: f, Angle: angle, Cos: cos, Sin: sin, CosTangent: ct, SinTangent: st})
				}
			}
		}
	}
	// Released geometry table [29,32,2] and both rotation directions. Use
	// deterministic bounded vectors, independent scalar convention already in CPU.
	rows, heads, dim := 29, 16, 64
	table := make([]float32, rows*dim)
	if err := train.RoPETableInto(table, rows, dim, 0, 10000); err != nil {
		t.Fatal(err)
	}
	for _, inverse := range []bool{false, true} {
		for pos := 0; pos < rows; pos++ {
			x := make([]float32, heads*dim)
			for i := range x {
				x[i] = float32(math.Sin(float64(i+1) * .13))
			}
			want := append([]float32(nil), x...)
			ropeScalar(want, pos, heads, dim, 10000, inverse)
			for head := 0; head < heads; head++ {
				for pair := 0; pair < dim/2; pair++ {
					at := head*dim + 2*pair
					c, s := table[pos*dim+2*pair], table[pos*dim+2*pair+1]
					if inverse {
						s = -s
					}
					a, b := x[at], x[at+1]
					x[at] = float32(a*c) - float32(b*s)
					x[at+1] = float32(b*c) + float32(a*s)
				}
			}
			for i := range x {
				d := math.Abs(float64(x[i]) - float64(want[i]))
				report.RopeValues++
				if d != 0 {
					report.RopeChanged++
				}
				report.RopeMaxError = max(report.RopeMaxError, d)
			}
		}
	}
	if report.MaxValueReferenceError > 6e-8 || report.MaxTangentReferenceError > 2e-5 || report.RopeMaxError > 1e-6 {
		t.Fatal("unchanged host boundary reference gates", report.MaxValueReferenceError, report.MaxTangentReferenceError, report.RopeMaxError)
	}
	if len(report.Rows) != 9984 || report.RopeValues != 59392 {
		t.Fatal("boundary inventory")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_SHARED_TIME_OUTPUT")
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(root, "time-boundary-inputs.json"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, we := f.Write(append(data, '\n'))
	ce := f.Close()
	if we != nil || ce != nil {
		t.Fatal(we, ce)
	}
	t.Logf("timeInputs=%d maxAngle=%g F64value=%g tangent=%g ropeValues=%d changed=%d max=%g", len(report.Rows), report.MaxAngle, report.MaxValueReferenceError, report.MaxTangentReferenceError, report.RopeValues, report.RopeChanged, report.RopeMaxError)
}
