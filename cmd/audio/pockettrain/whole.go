//go:build linux

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unicode/utf8"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

const preparedWholePolicy = "whole-utterance-shortest-distinct-train-reference-ceil1920-normalized-target-eos-pad1-v1"

// Whole manifests are derived training selections, not source checkpoint edits.
// Every row must explicitly be train and pin its admitted waveform. No dev row
// can enter target/reference pairing. Gitea sample manifests need a separately
// verified split conversion; absolute historical audio paths are rejected.
type wholeEntry struct {
	ID         string `json:"id"`
	Path       string `json:"audio"`
	Transcript string `json:"transcript"`
	SHA256     string `json:"source_sha256"`
	Split      string `json:"split"`
}

func readWholeManifest(data []byte, maxRows int) ([]wholeEntry, error) {
	if maxRows < 2 || maxRows > pockettts.MaxTrainingRunRows {
		return nil, fmt.Errorf("whole manifest needs2..100000 row ceiling")
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 4096), 64<<10)
	var rows []wholeEntry
	ids, paths, hashes := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for scanner.Scan() {
		if len(rows) >= maxRows {
			return nil, fmt.Errorf("whole manifest row ceiling")
		}
		// Reuse bounded duplicate-key admission; encoding/json alone silently
		// accepts conflicting split/hash declarations with last-value wins.
		if _, err := strictObject(scanner.Bytes()); err != nil {
			return nil, err
		}
		var e wholeEntry
		dec := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&e); err != nil {
			return nil, err
		}
		var extra any
		if dec.Decode(&extra) != io.EOF {
			return nil, fmt.Errorf("whole manifest trailing JSON")
		}
		if e.ID == "" || len(e.ID) > 256 || len(e.Path) > 4096 || !filepath.IsLocal(e.Path) || filepath.Clean(e.Path) != e.Path || e.Split != "train" || !validDigest(e.SHA256) || !utf8.ValidString(e.Transcript) || len(e.Transcript) > 4096 || strings.TrimSpace(e.Transcript) == "" || ids[e.ID] || paths[e.Path] || hashes[e.SHA256] {
			return nil, fmt.Errorf("invalid/duplicate whole training row %q", e.ID)
		}
		ids[e.ID], paths[e.Path], hashes[e.SHA256] = true, true, true
		rows = append(rows, e)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("whole manifest requires distinct target/reference training clips")
	}
	return rows, nil
}
func scanWholeRecording(root *os.Root, path string, maxSamples int) (int, asset, error) {
	f, err := root.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return 0, asset{}, err
	}
	samples, pin, scanErr := scanWAV(f, maxSamples)
	closeErr := f.Close()
	if scanErr != nil {
		return 0, asset{}, scanErr
	}
	if closeErr != nil {
		return 0, asset{}, closeErr
	}
	return samples, pin, nil
}
func readWholeWAV(root *os.Root, path string, pin asset, samples, maxSamples int) ([]float32, error) {
	f, err := root.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	n, actual, err := scanWAV(f, maxSamples)
	if err != nil {
		return nil, err
	}
	if actual != pin || n != samples {
		return nil, fmt.Errorf("reference waveform changed after admission")
	}
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}
	wav, rate, err := audio.ReadWAV(f)
	if err != nil {
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	if rate != 24000 || len(wav) != samples {
		return nil, fmt.Errorf("reference sample/rate mismatch")
	}
	return wav, nil
}
func assembleWholePreparedRow(target, reference []float32, row rowReport, tokens []uint32, mean, std []float32, encode func([]float32) ([]float32, int, error)) (pockettts.PreparedTrainingRow, error) {
	if len(target) != row.Samples || len(reference) != row.ReferenceSamples || len(target) == 0 || len(reference) == 0 || len(target) > 374*pockettts.SamplesPerFrame || len(reference) > 62*pockettts.SamplesPerFrame || row.TargetFrames != 1+(len(target)-1)/pockettts.SamplesPerFrame || row.VoiceFrames != 1+(len(reference)-1)/pockettts.SamplesPerFrame {
		return pockettts.PreparedTrainingRow{}, fmt.Errorf("whole-utterance sample/frame mismatch")
	}
	// Reuse checked assembly with two complete clips, not fabricated word cuts.
	// The private combined buffer has a synthetic split only for this internal
	// encoder seam; the persisted policy describes whole-utterance pairing.
	combined := make([]float32, 0, len(target)+len(reference))
	combined = append(combined, reference...)
	combined = append(combined, target...)
	geometry := rowReport{CutSample: len(reference), SegmentEndSample: len(combined), VoiceFrames: row.VoiceFrames, TargetFrames: row.TargetFrames}
	return assemblePreparedRow(combined, geometry, tokens, mean, std, encode)
}

func wholeReferences(samples []int) ([]int, error) {
	// Preserve all target utterances. Reference must itself be a complete clip
	// <=62 frames; choose the shortest distinct training clip, stable tie order.
	first, second := -1, -1
	for i, n := range samples {
		if n <= 0 || n > 374*pockettts.SamplesPerFrame {
			return nil, fmt.Errorf("target%d exceeds374valid frames; use reviewed splits/exclusions, no implicit crop", i)
		}
		if n > pockettts.ProductionTrainingVoiceFrames*pockettts.SamplesPerFrame {
			continue
		}
		if first < 0 || n < samples[first] {
			second = first
			first = i
		} else if second < 0 || n < samples[second] {
			second = i
		}
	}
	if second < 0 {
		return nil, fmt.Errorf("need two distinct complete training references within62frames")
	}
	out := make([]int, len(samples))
	for i := range out {
		out[i] = first
		if i == first {
			out[i] = second
		}
	}
	return out, nil
}
func preflightWholeDataset(r report, cfg pockettts.Config, tok *pockettts.Tokenizer, entries []wholeEntry, o options) (report, error) {
	root, err := os.OpenRoot(o.root)
	if err != nil {
		return r, err
	}
	defer root.Close()
	samples := make([]int, len(entries))
	pins := make([]asset, len(entries))
	for i, e := range entries {
		samples[i], pins[i], err = scanWholeRecording(root, e.Path, o.maxSamples)
		if err != nil {
			return r, fmt.Errorf("row%d: %w", i, err)
		}
		if pins[i].SHA256 != e.SHA256 {
			return r, fmt.Errorf("row%d waveform pin mismatch", i)
		}
	}
	refs, err := wholeReferences(samples)
	if err != nil {
		return r, err
	}
	for i, e := range entries {
		tokens, err := tok.Encode(e.Transcript)
		if err != nil {
			return r, err
		}
		if len(tokens) == 0 || len(tokens) > o.maxTokens {
			return r, fmt.Errorf("whole target text token ceiling")
		}
		frames := 1 + (samples[i]-1)/pockettts.SamplesPerFrame
		vf := 1 + (samples[refs[i]]-1)/pockettts.SamplesPerFrame
		plan, err := pockettts.PlanTrainingShape(cfg, pockettts.TrainingShape{MicroBatchRows: 1, TargetFrames: frames + 1, VoiceFrames: vf, TextTokens: len(tokens), GradientAccumulation: 1, FlowBatchMultiplier: 1}, pockettts.ProductionTrainingShapeLimits(o.maxResident))
		if err != nil {
			return r, err
		}
		r.Rows = append(r.Rows, rowReport{Index: i + 1, Path: e.Path, Audio: pins[i], Samples: samples[i], SegmentEndSample: samples[i], TextTokens: len(tokens), TargetFrames: frames, VoiceFrames: vf, SequenceRows: plan.SequenceRows, ResidentUpperBytes: plan.ResidentUpperBytes, ReferencePath: entries[refs[i]].Path, ReferenceAudio: pins[refs[i]], ReferenceSamples: samples[refs[i]]})
	}
	return r, nil
}
