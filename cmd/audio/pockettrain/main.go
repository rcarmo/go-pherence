//go:build linux

// pockettrain provides Linux CPU preflight, recording preparation and explicitly
// experimental SIMD or attention-backward hybrid training. Full GPU training
// is not implemented.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

type options struct {
	backend, config, model, tokenizer, manifest, root, modelHash, tokenizerHash string
	maxRows, maxSamples, maxTokens                                              int
	maxManifest, maxModel, maxResident                                          int64
	wholeManifest                                                               bool
	manifestHash                                                                string
}
type asset struct {
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}
type rowReport struct {
	ReferencePath      string  `json:"reference_path,omitempty"`
	ReferenceAudio     asset   `json:"reference_audio,omitempty"`
	ReferenceSamples   int     `json:"reference_samples,omitempty"`
	Index              int     `json:"index"`
	Path               string  `json:"path"`
	Audio              asset   `json:"audio"`
	Samples            int     `json:"samples"`
	CutSeconds         float64 `json:"cut_seconds"`
	SegmentStartSample int     `json:"segment_start_sample"`
	CutSample          int     `json:"cut_sample"`
	SegmentEndSample   int     `json:"segment_end_sample"`
	TextTokens         int     `json:"text_tokens"`
	TargetFrames       int     `json:"target_frames"`
	VoiceFrames        int     `json:"voice_frames"`
	SequenceRows       int     `json:"sequence_rows"`
	ResidentUpperBytes int64   `json:"estimated_training_resident_upper_bytes"`
}
type report struct {
	Mode                 string        `json:"mode"`
	Backend              string        `json:"backend"`
	TrainingImplemented  bool          `json:"training_command_implemented"`
	ModelValuesValidated bool          `json:"model_tensor_values_validated"`
	Config               asset         `json:"config"`
	Model                asset         `json:"model"`
	Tokenizer            asset         `json:"tokenizer"`
	Manifest             asset         `json:"manifest"`
	Rows                 []rowReport   `json:"rows"`
	Limits               optionsReport `json:"limits"`
	Notes                []string      `json:"notes"`
}
type optionsReport struct {
	MaxRows, MaxAudioSamples, MaxTextTokens                   int
	MaxManifestBytes, MaxModelBytes, MaxTrainingResidentBytes int64
}

func main() {
	var err error
	if importWholeRequested(os.Args[1:]) {
		err = importWholeMain(os.Args[1:], os.Stdout)
	} else if prepareRequested(os.Args[1:]) {
		err = prepareMain(os.Args[1:], os.Stdout)
	} else if trainingRequested(os.Args[1:]) {
		err = trainingMain(os.Args[1:], os.Stdout)
	} else {
		err = run(os.Args[1:], os.Stdout)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string, out io.Writer) error {
	var o options
	fs := flag.NewFlagSet("pockettrain", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	mode := fs.String("mode", "preflight", "read-only preflight; prepare/train dispatched separately")
	fs.StringVar(&o.backend, "backend", "simd", "simd; GPU training is unsupported")
	fs.StringVar(&o.config, "config", "", "Pocket English YAML")
	fs.StringVar(&o.model, "model", "", "model safetensors (header inspection and hash only)")
	fs.StringVar(&o.tokenizer, "tokenizer", "", "tokenizer JSON")
	fs.StringVar(&o.manifest, "manifest", "", "aligned JSONL manifest")
	fs.StringVar(&o.root, "source-root", "", "confined recording root; audio paths must be relative")
	fs.StringVar(&o.modelHash, "model-sha256", "", "required expected model SHA256")
	fs.StringVar(&o.tokenizerHash, "tokenizer-sha256", "", "required expected tokenizer SHA256")
	fs.IntVar(&o.maxRows, "max-rows", 1000, "maximum manifest rows")
	fs.IntVar(&o.maxSamples, "max-audio-samples", 35*24000, "maximum samples per recording")
	fs.IntVar(&o.maxTokens, "max-text-tokens", 512, "maximum target text tokens")
	fs.Int64Var(&o.maxManifest, "max-manifest-bytes", 8<<20, "maximum manifest file bytes")
	fs.Int64Var(&o.maxModel, "max-model-bytes", 1<<30, "maximum checkpoint file bytes")
	fs.Int64Var(&o.maxResident, "max-resident-bytes", 6<<30, "estimated training resident ceiling; excludes checkpoint peaks")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *mode != "preflight" {
		return fmt.Errorf("expected read-only preflight mode")
	}
	// Reject GPU requests before any file access or backend initialisation.
	if o.backend != "simd" {
		return fmt.Errorf("Pocket training backend %q unsupported: complete Vulkan/PTX loss/backward integration is not implemented", o.backend)
	}
	if o.config == "" || o.model == "" || o.tokenizer == "" || o.manifest == "" || o.root == "" || o.maxRows <= 0 || o.maxRows > 100000 || o.maxSamples <= 0 || o.maxSamples > 24000*3600 || o.maxTokens <= 0 || o.maxTokens > 512 || o.maxManifest <= 0 || o.maxManifest > 64<<20 || o.maxModel <= 0 || o.maxModel > 16<<30 || o.maxResident <= 0 {
		return fmt.Errorf("invalid preflight paths or explicit resource ceilings")
	}
	if !validDigest(o.modelHash) || !validDigest(o.tokenizerHash) {
		return fmt.Errorf("model-sha256 and tokenizer-sha256 must be lowercase SHA256 pins")
	}
	r, err := preflight(o)
	if err != nil {
		return err
	}
	return json.NewEncoder(out).Encode(r)
}
func validDigest(s string) bool {
	v, e := hex.DecodeString(s)
	return e == nil && len(v) == 32 && hex.EncodeToString(v) == s
}
func boundedFile(path string, limit int64) (*os.File, int64, error) {
	// Preliminary check avoids intentionally opening known devices. Descriptor
	// checks below still enforce admission after a raced path replacement.
	st, err := os.Stat(path)
	if err != nil {
		return nil, 0, err
	}
	if !st.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("%s is not a regular file", path)
	}
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, 0, err
	}
	st, err = f.Stat()
	if err != nil {
		f.Close()
		return nil, 0, fmt.Errorf("stat %s: %w", path, err)
	}
	if !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > limit {
		f.Close()
		return nil, 0, fmt.Errorf("%s is not a nonempty regular file within %d bytes", path, limit)
	}
	return f, st.Size(), nil
}
func boundedBytes(path string, limit int64) ([]byte, asset, error) {
	f, size, err := boundedFile(path, limit)
	if err != nil {
		return nil, asset{}, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, asset{}, err
	}
	if int64(len(data)) != size {
		return nil, asset{}, fmt.Errorf("%s changed size while reading", path)
	}
	h := sha256.Sum256(data)
	return data, asset{hex.EncodeToString(h[:]), size}, nil
}

// Source-only metadata seam for bounded synthetic CLI happy-path tests.
// Production always uses the strict metadata/hash reader; never loads tensors.
var preflightModelInfo = inspectModel

func preflight(o options) (report, error) {
	r := report{Mode: "preflight", Backend: o.backend, Limits: optionsReport{o.maxRows, o.maxSamples, o.maxTokens, o.maxManifest, o.maxModel, o.maxResident}, Notes: []string{"Read-only admission; no tensors materialised, codec encoding, gradients or optimiser updates.", "Checkpoint header/shape and digest validated; tensor finiteness and training execution untested.", "Resident estimate excludes checkpoint save/load peaks, filesystem free space and sustained training quality.", "First eligible aligned cut; timestamps round to nearest 24kHz sample and codec frames use ceiling(samples/1920). Prompt exceeding 62 frames (4.96s) is rejected, never trimmed; no resampling or clipping."}}
	data, a, err := boundedBytes(o.config, 1<<20)
	if err != nil {
		return r, err
	}
	r.Config = a
	cfg, err := pockettts.ParseConfig(data)
	if err != nil {
		return r, err
	}
	// Match the currently loadable released student, not resource-only teacher planning.
	c := cfg.FlowLM.Transformer
	if c.DModel != 1024 || c.NumLayers != 6 || c.NumHeads != 16 || c.HiddenScale != 4 || c.DimFeedforward != 0 || c.LayerScale != 0 || c.Context != 0 || c.MaxPeriod != 10000 || cfg.FlowLM.Flow.Type != "lsd" || cfg.FlowLM.Flow.Depth != 6 || cfg.FlowLM.Flow.Dim != 512 || cfg.FlowLM.LookupTable.NBins != 4000 || cfg.Mimi.InnerDim != 32 {
		return r, fmt.Errorf("preflight supports only the released six-layer English training topology")
	}
	r.Model, err = preflightModelInfo(o.model, o.maxModel, cfg)
	if err != nil {
		return r, err
	}
	if r.Model.SHA256 != o.modelHash {
		return r, fmt.Errorf("model digest mismatch")
	}
	data, a, err = boundedBytes(o.tokenizer, 1<<20)
	if err != nil {
		return r, err
	}
	r.Tokenizer = a
	if a.SHA256 != o.tokenizerHash {
		return r, fmt.Errorf("tokenizer digest mismatch")
	}
	tok, err := pockettts.LoadTokenizer(bytes.NewReader(data))
	if err != nil {
		return r, err
	}
	if tok.VocabSize() != cfg.FlowLM.LookupTable.NBins {
		return r, fmt.Errorf("tokenizer/config vocabulary mismatch")
	}
	data, a, err = boundedBytes(o.manifest, o.maxManifest)
	if err != nil {
		return r, err
	}
	r.Manifest = a
	if o.wholeManifest {
		if !validDigest(o.manifestHash) || a.SHA256 != o.manifestHash {
			return r, fmt.Errorf("whole-utterance source manifest pin mismatch")
		}
		entries, err := readWholeManifest(data, o.maxRows)
		if err != nil {
			return r, err
		}
		r.Notes = append(r.Notes, "Whole utterances override aligned-cut policy: complete targets and distinct shortest eligible train-only voice reference; no cropping, word timing or held-out conditioning.")
		return preflightWholeDataset(r, cfg, tok, entries, o)
	}
	entries, err := readManifest(data, o.maxRows)
	if err != nil {
		return r, err
	}
	return preflightDataset(r, cfg, tok, entries, o)
}

// Reject oversized transcript/word arrays before cut selection can construct
// quadratic suffix strings. The file and each raw line are already bounded.
func readManifest(data []byte, maxRows int) ([]pockettts.TrainingEntry, error) {
	s := bufio.NewScanner(bytes.NewReader(data))
	s.Buffer(make([]byte, 4096), 64<<10)
	rows := make([]pockettts.TrainingEntry, 0, min(maxRows, 1024))
	for s.Scan() {
		if len(rows) >= maxRows {
			return nil, fmt.Errorf("manifest row limit exceeded")
		}
		var e pockettts.TrainingEntry
		if err := json.Unmarshal(s.Bytes(), &e); err != nil {
			return nil, err
		}
		if len(e.Path) > 4096 || len(e.Transcript) > 4096 || len(e.Words) > 512 {
			return nil, fmt.Errorf("manifest path/transcript/word limit exceeded")
		}
		for _, w := range e.Words {
			if len(w.Word) > 128 {
				return nil, fmt.Errorf("manifest word byte limit exceeded")
			}
		}
		if err := e.Validate(); err != nil {
			return nil, err
		}
		rows = append(rows, e)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("manifest is empty")
	}
	return rows, nil
}

func preflightDataset(r report, cfg pockettts.Config, tok *pockettts.Tokenizer, entries []pockettts.TrainingEntry, o options) (report, error) {
	root, err := os.OpenRoot(o.root)
	if err != nil {
		return r, err
	}
	defer root.Close()
	r.Rows = make([]rowReport, 0, len(entries))
	for index, e := range entries {
		if !filepath.IsLocal(e.Path) {
			return r, fmt.Errorf("row %d: audio path must be relative within source-root", index+1)
		}
		st, err := root.Stat(e.Path)
		if err != nil {
			return r, fmt.Errorf("row %d: %w", index+1, err)
		}
		if !st.Mode().IsRegular() {
			return r, fmt.Errorf("row %d: audio is not a regular file", index+1)
		}
		f, err := root.OpenFile(e.Path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return r, fmt.Errorf("row %d: %w", index+1, err)
		}
		samples, provenance, err := scanWAV(f, o.maxSamples)
		closeErr := f.Close()
		if err != nil {
			return r, fmt.Errorf("row %d: %w", index+1, err)
		}
		if closeErr != nil {
			return r, closeErr
		}
		row, err := planRow(cfg, tok, e, samples, o.maxTokens, o.maxResident)
		if err != nil {
			return r, fmt.Errorf("row %d: %w", index+1, err)
		}
		row.Index = index + 1
		row.Path = e.Path
		row.Audio = provenance
		r.Rows = append(r.Rows, row)
	}
	return r, nil
}
