//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	asr "github.com/rcarmo/go-pherence/model/nemotronasr"
)

const NemotronASRRevision = "ea30d66debe3740a08b573244286791d423d6b3e"
const NemotronEmissionTiming = "rnnt-emission-chunks-not-word-alignment-v1"

// NemotronASRConfig identifies the loaded assets and explicitly selected prompt.
// Automatic prompting does not report a detected language; Language stays und.
type NemotronASRConfig struct {
	ModelSHA256, TokenizerSHA256, RuntimeSHA256, ModelRevision string
	Language                                                   string
	PromptID                                                   int
	MaxResultBytes                                             int64
}

type nemotronASRStream interface {
	AppendPCM(context.Context, []float32) ([]int, []int64, error)
	Finish(context.Context) ([]int, []int64, error)
}

// NewNemotronASRStage publishes transcript directly. A fresh CPU stream is used
// per attempt; retry reuses verified completed decode/other stages, but replays
// this ASR stage from zero after cancellation. No GPU or silent fallback occurs.
// Cue timing is explicitly coarse emission/input chunks, never word alignment.
func NewNemotronASRStage(model *asr.PCMGenerationModel, vocab *tokenizer.Tokenizer, cfg NemotronASRConfig) (Stage, error) {
	if model == nil || model.Subsampling == nil || model.Tower == nil || model.Projection == nil || model.Decoder == nil {
		return Stage{}, ErrConfiguration
	}
	return newNemotronASRStage(vocab, cfg, func() nemotronASRStream {
		prompt := cfg.PromptID
		return &asr.PCMGenerationStream{Model: model, PromptID: &prompt}
	})
}

func newNemotronASRStage(vocab *tokenizer.Tokenizer, cfg NemotronASRConfig, factory func() nemotronASRStream) (Stage, error) {
	if vocab == nil || len(vocab.InvVocab) < 13087 || !validHash(cfg.ModelSHA256) || !validHash(cfg.TokenizerSHA256) || !validHash(cfg.RuntimeSHA256) || cfg.ModelRevision != NemotronASRRevision || cfg.PromptID < 0 || cfg.PromptID >= 128 || cfg.MaxResultBytes < 1 || cfg.MaxResultBytes > MaxTranscriptBytes || factory == nil {
		return Stage{}, ErrConfiguration
	}
	expected := map[string]int{"auto": 101, "en": 0, "pt": 13, "fr": 8, "es": 2, "it": 15}
	if prompt, ok := expected[cfg.Language]; !ok || prompt != cfg.PromptID {
		return Stage{}, ErrConfiguration
	}
	identity, _ := json.Marshal(struct {
		Schema string
		Config NemotronASRConfig
	}{"speechjob-nemotron-asr-cpu-emission-chunks-v1", cfg})
	return Stage{Name: "transcript", Version: hash(identity), Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "decode" {
				decoded = cp.Blob
			}
		}
		if decoded.File == "" {
			return ErrConfiguration
		}
		r, err := in.store.openBlob(ctx, in.job.ID, decoded)
		if err != nil {
			return err
		}
		if err = r.Close(); err != nil {
			return err
		}
		pcm, err := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, pcm.Close()) }()
		total := int64(pcm.Timeline().Samples)
		if total < 160 || total > 4*3600*16000 {
			return ErrLimit
		}
		used, _, err := in.store.usage()
		if err != nil {
			return err
		}
		if cfg.MaxResultBytes > in.store.limits.MaxArtifactBytes || cfg.MaxResultBytes+maxManifest > in.store.limits.MaxBytes-used {
			return ErrLimit
		}
		stream := factory()
		if stream == nil {
			return ErrConfiguration
		}
		language := cfg.Language
		if language == "auto" {
			language = "und"
		}
		text := Transcript{Schema: 2, SampleRate: 16000, TotalSamples: total, Language: language, SourceTiming: pcm.SourceTiming(), Cues: []Cue{}, Timing: NemotronEmissionTiming}
		pending := ""
		var textBytes int64
		var cueStart, lastFrame int64
		lastFrame = -1
		// Split only at actual decoder whitespace so subwords across PCM calls
		// remain intact. Delayed emission is placed in coarse, labelled chunks.
		consume := func(tokens []int, frames []int64, end int64, final bool) error {
			if len(tokens) != len(frames) {
				return ErrCorrupt
			}
			ids := make([]int, 0, len(tokens))
			for i, t := range tokens {
				if frames[i] < 0 || frames[i] < lastFrame || frames[i] > total/1280+4 || t < 0 || t > 13087 {
					return ErrCorrupt
				}
				lastFrame = frames[i]
				if t == 13087 {
					continue
				}
				piece, ok := vocab.InvVocab[t]
				if !ok {
					return ErrCorrupt
				}
				if _, special := vocab.AddedSpecial[piece]; !special {
					ids = append(ids, t)
				}
			}
			pending += vocab.Decode(ids)
			if int64(len(pending))+textBytes > cfg.MaxResultBytes {
				return ErrLimit
			}
			split := len(pending)
			if !final {
				split = strings.LastIndexFunc(pending, unicode.IsSpace)
				if split < 0 {
					return nil
				}
			}
			chunk := strings.TrimSpace(pending[:split])
			pending = pending[split:]
			if chunk != "" {
				if len(chunk) > 65536 || len(text.Cues) >= MaxTranscriptCues {
					return ErrLimit
				}
				if cueStart >= end && len(text.Cues) > 0 {
					text.Cues[len(text.Cues)-1].Text += " " + chunk
					textBytes += int64(len(chunk)) + 1
					return ctx.Err()
				}
				start := min(cueStart, total-1)
				end = max(start+1, min(end, total))
				text.Cues = append(text.Cues, Cue{StartSample: start, EndSample: end, Speaker: -1, Text: chunk})
				textBytes += int64(len(chunk))
				cueStart = end
			}
			return ctx.Err()
		}
		scratch := make([]float32, 80000)
		reportWorkProgress(ctx, "asr-windows", 0, total, "samples")
		for offset := int64(0); offset < total; {
			if err := ctx.Err(); err != nil {
				return err
			}
			count := int(min(int64(len(scratch)), total-offset))
			n, err := pcm.ReadSamplesAt(ctx, scratch[:count], offset)
			if err != nil && !errors.Is(err, io.EOF) {
				return err
			}
			if n != count {
				return io.ErrUnexpectedEOF
			}
			tokens, frames, err := stream.AppendPCM(ctx, scratch[:count])
			if err != nil {
				return err
			}
			offset += int64(count)
			if err = consume(tokens, frames, offset, false); err != nil {
				return err
			}
			reportWorkProgress(ctx, "asr-windows", offset, total, "samples")
		}
		tokens, frames, err := stream.Finish(ctx)
		if err != nil {
			return err
		}
		if err = consume(tokens, frames, total, true); err != nil {
			return err
		}
		var data bytes.Buffer
		if err = WriteTranscriptJSON(ctx, &data, text); err != nil {
			return err
		}
		if int64(data.Len()) > cfg.MaxResultBytes {
			return ErrLimit
		}
		return writeContext(ctx, out, data.Bytes())
	}}, nil
}
