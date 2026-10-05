//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"syscall"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

const preparedPolicy = "aligned-first-cut-nearest24k-ceil1920-normalized-target-eos-pad1-v1"

func prepareRequested(args []string) bool {
	for i, a := range args {
		if a == "-mode=prepare" || a == "--mode=prepare" || (a == "-mode" || a == "--mode") && i+1 < len(args) && args[i+1] == "prepare" {
			return true
		}
	}
	return false
}
func prepareMain(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("pockettrain prepare", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	mode := fs.String("mode", "prepare", "prepare")
	backend := fs.String("backend", "simd", "SIMD only")
	config := fs.String("config", "", "released English YAML")
	model := fs.String("model", "", "released model including Mimi encoder")
	modelHash := fs.String("model-sha256", "", "required model pin")
	tokenizer := fs.String("tokenizer", "", "tokenizer JSON")
	tokenizerHash := fs.String("tokenizer-sha256", "", "required tokenizer pin")
	manifest := fs.String("manifest", "", "aligned JSONL or derived whole-utterance training manifest")
	manifestFormat := fs.String("manifest-format", "aligned", "aligned or whole-utterance")
	manifestHash := fs.String("manifest-sha256", "", "required for derived whole-utterance selection")
	sourceRoot := fs.String("source-root", "", "recording root")
	dest := fs.String("prepared-output", "", "new prepared JSON destination")
	maxRows := fs.Int("max-rows", 1000, "row ceiling")
	maxBytes := fs.Int64("max-prepared-bytes", 64<<20, "output byte ceiling")
	maxResident := fs.Int64("max-resident-bytes", 6<<30, "preflight training-memory admission")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *backend != "simd" {
		return fmt.Errorf("GPU preparation/training unavailable")
	}
	if *mode != "prepare" || fs.NArg() != 0 || *dest == "" || *maxBytes <= 0 || *maxBytes > 256<<20 || (*manifestFormat != "aligned" && *manifestFormat != "whole-utterance") {
		return fmt.Errorf("invalid prepare options")
	}
	if _, err := os.Lstat(*dest); err == nil || !os.IsNotExist(err) {
		return fmt.Errorf("prepared-output must not exist")
	}
	o := options{backend: *backend, config: *config, model: *model, tokenizer: *tokenizer, manifest: *manifest, root: *sourceRoot, modelHash: *modelHash, tokenizerHash: *tokenizerHash, maxRows: *maxRows, maxSamples: 35 * 24000, maxTokens: 512, maxManifest: 8 << 20, maxModel: 1 << 30, maxResident: *maxResident}
	o.wholeManifest = *manifestFormat == "whole-utterance"
	o.manifestHash = *manifestHash
	if o.wholeManifest {
		if err := requireDerivedOutput(*sourceRoot, *dest); err != nil {
			return err
		}
	}
	if o.wholeManifest && !validDigest(*manifestHash) {
		return fmt.Errorf("whole-utterance selection requires manifest-sha256")
	}
	if !validDigest(*modelHash) || !validDigest(*tokenizerHash) || *maxRows <= 0 || *maxRows > 100000 || *maxResident <= 0 {
		return fmt.Errorf("prepare requires pins and positive ceilings")
	}
	report, err := preflight(o)
	if err != nil {
		return err
	}
	cfgData, cfgPin, err := boundedBytes(*config, 1<<20)
	if err != nil {
		return err
	}
	cfg, err := pockettts.ParseConfig(cfgData)
	if err != nil {
		return err
	}
	if cfgPin != report.Config {
		return fmt.Errorf("config changed after preflight")
	}
	tokData, tokPin, err := boundedBytes(*tokenizer, 1<<20)
	if err != nil {
		return err
	}
	tok, err := pockettts.LoadTokenizer(bytes.NewReader(tokData))
	if err != nil {
		return err
	}
	if tokPin != report.Tokenizer {
		return fmt.Errorf("tokenizer changed after preflight")
	}
	manifestData, manifestPin, err := boundedBytes(*manifest, 8<<20)
	if err != nil {
		return err
	}
	if manifestPin != report.Manifest {
		return fmt.Errorf("manifest changed after preflight")
	}
	var entries []pockettts.TrainingEntry
	var whole []wholeEntry
	if o.wholeManifest {
		whole, err = readWholeManifest(manifestData, *maxRows)
	} else {
		entries, err = readManifest(manifestData, *maxRows)
	}
	if err != nil {
		return err
	}
	file, size, err := boundedFile(*model, 1<<30)
	if err != nil {
		return err
	}
	defer file.Close()
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(file, size+1))
	if err != nil || n != size || hex.EncodeToString(h.Sum(nil)) != *modelHash {
		return fmt.Errorf("model changed after preflight")
	}
	source, err := safetensors.Open(fmt.Sprintf("/proc/self/fd/%d", file.Fd()))
	if err != nil {
		return err
	}
	defer source.Close()
	encoder, err := pockettts.LoadMimiEncoderCPU(source, cfg)
	if err != nil {
		return err
	}
	mean, _, err := source.GetFloat32("flow_lm.emb_mean")
	if err != nil {
		return err
	}
	std, _, err := source.GetFloat32("flow_lm.emb_std")
	if err != nil {
		return err
	}
	if len(mean) != 32 || len(std) != 32 {
		return fmt.Errorf("latent stats shape")
	}
	for i, x := range std {
		if x <= 0 || math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(mean[i])) || math.IsInf(float64(mean[i]), 0) {
			return fmt.Errorf("invalid latent statistics")
		}
	}
	root, err := os.OpenRoot(*sourceRoot)
	if err != nil {
		return err
	}
	defer root.Close()
	corpus := preparedCorpus{Version: 1, Source: &preparedProvenance{ModelSHA256: report.Model.SHA256, ConfigSHA256: report.Config.SHA256, TokenizerSHA256: report.Tokenizer.SHA256, ManifestSHA256: report.Manifest.SHA256, Policy: preparedPolicy}}
	if o.wholeManifest {
		corpus.Source.Policy = preparedWholePolicy
	}
	retainedBytes := int64(0)
	for i, row := range report.Rows {
		e := pockettts.TrainingEntry{Path: row.Path}
		if !o.wholeManifest {
			e = entries[i]
		}
		if !filepath.IsLocal(e.Path) {
			return fmt.Errorf("recording path outside root")
		}
		f, err := root.OpenFile(e.Path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return err
		}
		_, provenance, err := scanWAV(f, o.maxSamples)
		if err != nil {
			f.Close()
			return err
		}
		if provenance.SHA256 != report.Rows[i].Audio.SHA256 {
			f.Close()
			return fmt.Errorf("recording changed after preflight")
		}
		if _, err = f.Seek(0, 0); err != nil {
			f.Close()
			return err
		}
		wav, rate, readErr := audio.ReadWAV(f)
		closeErr := f.Close()
		if readErr != nil {
			return readErr
		}
		if closeErr != nil {
			return closeErr
		}
		if rate != 24000 {
			return fmt.Errorf("recording rate changed")
		}
		if row.TargetFrames >= 375 {
			return fmt.Errorf("row%d: EOS padding would exceed375 physical frames", i)
		}
		physical := pockettts.TrainingShape{MicroBatchRows: 1, TargetFrames: row.TargetFrames + 1, VoiceFrames: row.VoiceFrames, TextTokens: row.TextTokens, GradientAccumulation: 1, FlowBatchMultiplier: 1}
		if _, err := pockettts.PlanTrainingShape(cfg, physical, pockettts.ProductionTrainingShapeLimits(*maxResident)); err != nil {
			return fmt.Errorf("row%d padded shape: %w", i, err)
		}
		var prepared pockettts.PreparedTrainingRow
		var tokens []uint32
		if o.wholeManifest {
			if len(wav) != row.Samples {
				return fmt.Errorf("whole recording sample count changed")
			}
			reference, err := readWholeWAV(root, row.ReferencePath, row.ReferenceAudio, row.ReferenceSamples, o.maxSamples)
			if err != nil {
				return err
			}
			tokens, err = tok.Encode(whole[i].Transcript)
			if err != nil {
				return err
			}
			prepared, err = assembleWholePreparedRow(wav, reference, row, tokens, mean, std, encoder.Encode)
			if err != nil {
				return fmt.Errorf("row%d: %w", i, err)
			}
		} else {
			if row.CutSample <= row.SegmentStartSample || row.SegmentEndSample > len(wav) {
				return fmt.Errorf("cut sample bounds")
			}
			cuts := e.EligibleTrainingCuts(pockettts.TrainingMinimumCutSeconds)
			tokens, err = tok.Encode(cuts[0].TargetTranscript)
			if err != nil {
				return err
			}
			prepared, err = assemblePreparedRow(wav, row, tokens, mean, std, encoder.Encode)
			if err != nil {
				return fmt.Errorf("row%d: %w", i, err)
			}
		}
		retainedBytes += int64(len(prepared.Batch.NormalizedLatents)+len(prepared.Batch.VoiceLatents)+len(tokens)) * 4
		if retainedBytes > *maxBytes/4 || retainedBytes > *maxResident/8 {
			return fmt.Errorf("prepared corpus retained-memory/serialized ceiling")
		}
		corpus.Rows = append(corpus.Rows, prepared)
	}
	parent := filepath.Dir(*dest)
	temp, err := os.CreateTemp(parent, ".prepared-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()
	w := &preparedLimitWriter{writer: temp, left: *maxBytes}
	if err := json.NewEncoder(w).Encode(corpus); err != nil {
		return err
	}
	if err := temp.Sync(); err != nil {
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Link(temp.Name(), *dest); err != nil {
		return err
	}
	d, err := os.Open(parent)
	if err != nil {
		return fmt.Errorf("prepared output visible; directory sync uncertain: %w", err)
	}
	syncErr := d.Sync()
	closeErr := d.Close()
	if syncErr != nil || closeErr != nil {
		return fmt.Errorf("prepared output visible; directory sync/close uncertain: %v %v", syncErr, closeErr)
	}
	return json.NewEncoder(out).Encode(map[string]any{"mode": "prepare", "rows": len(corpus.Rows), "output": *dest, "policy": corpus.Source.Policy, "training_executed": false})
}

// The codec function is supplied directly, not a mutable global test hook.
// Synthetic tests can inspect sample ranges and row assembly independently.
func assemblePreparedRow(wav []float32, row rowReport, tokens []uint32, mean, std []float32, encode func([]float32) ([]float32, int, error)) (pockettts.PreparedTrainingRow, error) {
	var result pockettts.PreparedTrainingRow
	channels := len(mean)
	if encode == nil || channels == 0 || len(std) != channels || len(tokens) == 0 || len(tokens) > 512 || row.SegmentStartSample < 0 || row.CutSample <= row.SegmentStartSample || row.SegmentEndSample <= row.CutSample || row.SegmentEndSample > len(wav) || row.TargetFrames <= 0 || row.TargetFrames >= 375 || row.VoiceFrames <= 0 || row.VoiceFrames > 62 {
		return result, fmt.Errorf("invalid prepared row geometry/EOS ceiling")
	}
	voice, vframes, err := encode(wav[row.SegmentStartSample:row.CutSample])
	if err != nil {
		return result, err
	}
	voice = append([]float32(nil), voice...) // Codec output may use reusable storage.
	target, frames, err := encode(wav[row.CutSample:row.SegmentEndSample])
	if err != nil {
		return result, err
	}
	if frames != row.TargetFrames || vframes != row.VoiceFrames || len(voice) != vframes*channels || len(target) != frames*channels {
		return result, fmt.Errorf("codec/preflight frame or channel mismatch")
	}
	for _, x := range voice {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return result, fmt.Errorf("nonfinite voice latent")
		}
	}
	normalized, err := normalizePrepared(target, mean, std)
	if err != nil {
		return result, err
	}
	normalized = append(normalized, make([]float32, channels)...)
	result = pockettts.PreparedTrainingRow{Batch: pockettts.FlowLMTrainingBatch{Frames: frames + 1, VoiceFrames: vframes, NormalizedLatents: normalized, VoiceLatents: append([]float32(nil), voice...), TextTokens: append([]uint32(nil), tokens...)}, ValidFrames: frames}
	return result, nil
}

func normalizePrepared(target, mean, std []float32) ([]float32, error) {
	if len(target) == 0 || len(mean) == 0 || len(mean) != len(std) || len(target)%len(mean) != 0 {
		return nil, fmt.Errorf("latent normalization shape")
	}
	out := make([]float32, len(target))
	for i, x := range target {
		c := i % len(mean)
		if std[c] <= 0 || math.IsNaN(float64(std[c])) || math.IsInf(float64(std[c]), 0) || math.IsNaN(float64(mean[c])) || math.IsInf(float64(mean[c]), 0) {
			return nil, fmt.Errorf("latent std must be positive")
		}
		out[i] = (x - mean[c]) / std[c]
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			return nil, fmt.Errorf("normalized latent nonfinite")
		}
	}
	return out, nil
}

type preparedLimitWriter struct {
	writer io.Writer
	left   int64
}

func (w *preparedLimitWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.left {
		return 0, fmt.Errorf("prepared output byte ceiling")
	}
	n, err := w.writer.Write(p)
	w.left -= int64(n)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	return n, err
}
