//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Offline input-boundary diagnostic only: emulate the GLSL precise sequential
// F32 reduction against actual SIMD dispatch and an independent compensated F64
// dot product. No Vulkan interpreter, runtime or shader execution occurs here.
// This cannot locate the cause of the saved released gradient/update failure.
func TestPocketSharedReleasedSpeakerReductionDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_PROJECTION_DIAGNOSTIC") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_PROJECTION_DIAGNOSTIC=1 with pinned model/prepared corpus")
	}
	path := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	corpusPath := os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS")
	info, err := os.Stat(corpusPath)
	if err != nil || info.Size() > 1<<20 {
		t.Fatal("prepared corpus size", err)
	}
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9" {
		t.Fatal("prepared corpus provenance mismatch")
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal("prepared corpus", err)
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	weight, shape, err := file.GetFloat32("flow_lm.speaker_proj_weight")
	if err != nil || len(shape) != 2 || shape[0] != 1024 || shape[1] != 32 {
		t.Fatal("released speaker projection", err, shape)
	}
	batch := corpus.Rows[0].Batch
	if batch.VoiceFrames != 13 || len(batch.VoiceLatents) != batch.VoiceFrames*shape[1] || !finiteF32(batch.VoiceLatents) {
		t.Fatal("pinned voice rows")
	}
	linear := LinearF32{In: shape[1], Out: shape[0], Weight: weight}
	var cpuAbs, shaderAbs, betweenAbs, cpuSquare, shaderSquare, betweenSquare float64
	var cpuMax, shaderMax, betweenMax float64
	var differences, signs int
	var worstRow, worstCol int
	var worstCPU, worstShader float32
	var worstReference float64
	for row := 0; row < batch.VoiceFrames; row++ {
		input := batch.VoiceLatents[row*linear.In : (row+1)*linear.In]
		cpu := make([]float32, linear.Out)
		// Explicit dispatched CPU path, with no fallback on an error.
		if err := AffineSIMD(cpu, input, weight, nil, linear.In, linear.Out); err != nil {
			t.Fatal(err)
		}
		for col := 0; col < linear.Out; col++ {
			var shader float32
			var reference, correction float64
			for q := 0; q < linear.In; q++ {
				w := weight[col*linear.In+q]
				// Explicit conversion prohibits Go contraction across the F32 product.
				shader += float32(input[q] * w)
				term := float64(input[q])*float64(w) - correction
				next := reference + term
				correction = (next - reference) - term
				reference = next
			}
			if !isFinite(shader) || !isFinite(cpu[col]) || !finite64(reference) {
				t.Fatal("nonfinite boundary")
			}
			c, s, d := math.Abs(float64(cpu[col])-reference), math.Abs(float64(shader)-reference), math.Abs(float64(cpu[col])-float64(shader))
			cpuAbs += c
			shaderAbs += s
			betweenAbs += d
			cpuSquare += c * c
			shaderSquare += s * s
			betweenSquare += d * d
			cpuMax, shaderMax = math.Max(cpuMax, c), math.Max(shaderMax, s)
			if d > betweenMax {
				betweenMax = d
				worstRow, worstCol, worstCPU, worstShader, worstReference = row, col, cpu[col], shader, reference
			}
			if d > 0 {
				differences++
			}
			if cpu[col]*shader < 0 {
				signs++
			}
		}
	}
	n := float64(batch.VoiceFrames * linear.Out)
	t.Logf("offline speaker forward before transformer: rows=%d in=%d out=%d values=%.0f CPU-vs-GLSL-emulation different=%d oppositeSigns=%d MAE=%g RMS=%g max=%g", batch.VoiceFrames, linear.In, linear.Out, n, differences, signs, betweenAbs/n, math.Sqrt(betweenSquare/n), betweenMax)
	t.Logf("independent compensated F64 reference: CPU MAE=%g RMS=%g max=%g; GLSL-emulation MAE=%g RMS=%g max=%g", cpuAbs/n, math.Sqrt(cpuSquare/n), cpuMax, shaderAbs/n, math.Sqrt(shaderSquare/n), shaderMax)
	t.Logf("worst between-path point: row=%d col=%d CPU=%g GLSL-emulation=%g F64-reference=%.17g", worstRow, worstCol, worstCPU, worstShader, worstReference)
}
