package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

const MaxTranscriptBytes = 16 << 20
const MaxTranscriptCues = 100000

// Transcript is a versioned, model-neutral final transcript, not raw overlapping
// inference windows. All timestamps are integer canonical16k samples. Callers
// must reconcile windows and align speakers before publication; these serializers
// do not invent word timing, perform deduplication or identify speaker voices.
// Speaker=-1 denotes unlabelled text;0..63 are already assigned local job IDs.
// Cues may overlap, but starts are nondecreasing and all spans lie within input.
// Words are an independent checked timeline because cross-attention boundaries
// need not be contained by Whisper segment timestamp boundaries.
type Transcript struct {
	Schema       int                `json:"schema"`
	SampleRate   int                `json:"sample_rate"`
	TotalSamples int64              `json:"total_samples"`
	Language     string             `json:"language"`
	SourceTiming media.SourceTiming `json:"source_timing"`
	Cues         []Cue              `json:"cues"`
	Words        []WordCue          `json:"words,omitempty"`
}
type Cue struct {
	StartSample int64  `json:"start_sample"`
	EndSample   int64  `json:"end_sample"`
	Speaker     int    `json:"speaker"`
	Text        string `json:"text"`
}

// WordCue is a checked word boundary on the canonical 16 kHz sample timeline.
// Speaker=-1 means that exclusive diarization did not uniquely cover the word.
type WordCue struct {
	StartSample int64  `json:"start_sample"`
	EndSample   int64  `json:"end_sample"`
	Speaker     int    `json:"speaker"`
	Text        string `json:"text"`
}

func validateTranscript(ctx context.Context, t Transcript) error {
	if e := ctx.Err(); e != nil {
		return e
	}
	if t.Schema != 2 || t.SampleRate != 16000 || t.TotalSamples < 0 || t.TotalSamples > 4*3600*16000 || len(t.Cues) > MaxTranscriptCues || len(t.Language) < 2 || len(t.Language) > 32 {
		return fmt.Errorf("invalid transcript schema/geometry/language")
	}
	if _, e := media.MarshalSourceTimingWAVChunk(t.SourceTiming); e != nil {
		return fmt.Errorf("invalid transcript source timing: %w", e)
	}
	for _, c := range t.Language {
		if !(c >= 'a' && c <= 'z' || c == '-') {
			return fmt.Errorf("invalid transcript language")
		}
	}
	var previous int64
	totalText := 0
	for i, c := range t.Cues {
		if i%256 == 0 {
			if e := ctx.Err(); e != nil {
				return e
			}
		}
		if c.StartSample < 0 || c.EndSample <= c.StartSample || c.EndSample > t.TotalSamples || i > 0 && c.StartSample < previous || c.Speaker < -1 || c.Speaker > 63 || len(c.Text) > 65536 || !utf8.ValidString(c.Text) || strings.TrimSpace(c.Text) == "" {
			return fmt.Errorf("invalid transcript cue%d", i)
		}
		for _, r := range c.Text {
			if unicode.IsControl(r) && r != '\n' && r != '\t' {
				return fmt.Errorf("invalid control character in cue%d", i)
			}
		}
		totalText += len(c.Text)
		if totalText > MaxTranscriptBytes {
			return ErrLimit
		}
		previous = c.StartSample
	}
	previous = 0
	for i, word := range t.Words {
		if word.StartSample < 0 || word.EndSample < word.StartSample || word.EndSample > t.TotalSamples || i > 0 && word.StartSample < previous || word.Speaker < -1 || word.Speaker > 63 || len(word.Text) == 0 || len(word.Text) > 65536 || !utf8.ValidString(word.Text) || strings.TrimSpace(word.Text) == "" {
			return fmt.Errorf("invalid transcript word%d", i)
		}
		for _, r := range word.Text {
			if unicode.IsControl(r) {
				return fmt.Errorf("invalid control character in word%d", i)
			}
		}
		totalText += len(word.Text)
		if totalText > MaxTranscriptBytes {
			return ErrLimit
		}
		previous = word.StartSample
	}
	return ctx.Err()
}

// WriteTranscriptJSON validates the whole document before writing. Empty cues
// are canonicalised to [], not null. An IO/cancellation error can leave a partial
// writer; Store.Run only publishes it after successful completion.
func WriteTranscriptJSON(ctx context.Context, w io.Writer, t Transcript) error {
	if w == nil {
		return fmt.Errorf("nil transcript writer")
	}
	if e := validateTranscript(ctx, t); e != nil {
		return e
	}
	if t.Cues == nil {
		t.Cues = []Cue{}
	}
	b, e := json.Marshal(t)
	if e != nil {
		return e
	}
	b = append(b, '\n')
	if len(b) > MaxTranscriptBytes {
		return ErrLimit
	}
	return writeContext(ctx, w, b)
}
func ReadTranscriptJSON(ctx context.Context, r io.Reader) (Transcript, error) {
	var t Transcript
	if e := ctx.Err(); e != nil {
		return t, e
	}
	if r == nil {
		return t, fmt.Errorf("nil transcript reader")
	}
	var b bytes.Buffer
	buf := make([]byte, 32<<10)
	for {
		if e := ctx.Err(); e != nil {
			return t, e
		}
		n, e := r.Read(buf)
		if n > 0 {
			if b.Len()+n > MaxTranscriptBytes {
				return t, ErrLimit
			}
			b.Write(buf[:n])
		}
		if e == io.EOF {
			break
		}
		if e != nil {
			return t, e
		}
		if n == 0 {
			return t, io.ErrNoProgress
		}
	}
	if !utf8.Valid(b.Bytes()) {
		return Transcript{}, fmt.Errorf("invalid UTF-8 transcript JSON")
	}
	if e := validateTranscriptJSONShape(ctx, b.Bytes()); e != nil {
		return Transcript{}, e
	}
	d := json.NewDecoder(&b)
	d.DisallowUnknownFields()
	if e := d.Decode(&t); e != nil {
		return Transcript{}, e
	}
	if d.Decode(new(any)) != io.EOF {
		return Transcript{}, fmt.Errorf("trailing transcript JSON")
	}
	if e := validateTranscript(ctx, t); e != nil {
		return Transcript{}, e
	}
	return t, nil
}
func writeContext(ctx context.Context, w io.Writer, b []byte) error {
	for len(b) > 0 {
		if e := ctx.Err(); e != nil {
			return e
		}
		n := min(len(b), 32<<10)
		if e := writeFull(w, b[:n]); e != nil {
			return e
		}
		b = b[n:]
	}
	return ctx.Err()
}
func vttTime(ms int64) string {
	return fmt.Sprintf("%02d:%02d:%02d.%03d", ms/3600000, ms/60000%60, ms/1000%60, ms%1000)
}

// WriteWebVTT rounds starts down and ends up to milliseconds, preserving even
// sub-ms spans without zero-length cues. Checked zero-duration word spans are
// projected to one millisecond because WebVTT requires end>start; JSON retains
// their exact equal sample boundaries. Rounding can add <1ms to each boundary.
// Text is escaped (including '<', '&', '-->') before a generated voice tag; no
// input HTML/VTT markup is trusted. Newlines/tabs become spaces to prevent blank
// line injection of new cues or metadata. Empty transcripts produce WEBVTT only.
func WriteWebVTT(ctx context.Context, w io.Writer, t Transcript) error {
	if w == nil {
		return fmt.Errorf("nil VTT writer")
	}
	if e := validateTranscript(ctx, t); e != nil {
		return e
	}
	return writeVTTCues(ctx, w, t.Cues)
}

// writeVTTCues renders already-validated cue-like spans. Speaker-word output
// uses this after validating WordCue records, which may have equal boundaries.
func writeVTTCues(ctx context.Context, w io.Writer, cues []Cue) error {
	var b bytes.Buffer
	b.WriteString("WEBVTT\n\n")
	for i, c := range cues {
		if i%256 == 0 {
			if e := ctx.Err(); e != nil {
				return e
			}
		}
		b.WriteString(strconv.Itoa(i + 1))
		b.WriteByte('\n')
		startMS := c.StartSample / 16
		endMS := (c.EndSample + 15) / 16
		// Checked word alignment can legitimately put adjacent token boundaries
		// on the same 20 ms frame. JSON preserves that zero-duration span; VTT
		// requires end>start, so project it to the smallest representable cue.
		if endMS <= startMS {
			endMS = startMS + 1
		}
		b.WriteString(vttTime(startMS))
		b.WriteString(" --> ")
		b.WriteString(vttTime(endMS))
		b.WriteByte('\n')
		if c.Speaker >= 0 {
			fmt.Fprintf(&b, "<v SPEAKER_%02d>", c.Speaker)
		}
		// WebVTT supports named &amp;/&lt;/&gt; references, not HTML's numeric
		// quote references. Quotes are safe as literal cue text.
		text := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", "\n", " ", "\t", " ").Replace(c.Text)
		b.WriteString(text)
		if c.Speaker >= 0 {
			b.WriteString("</v>")
		}
		b.WriteString("\n\n")
		if b.Len() > MaxTranscriptBytes {
			return ErrLimit
		}
	}
	return writeContext(ctx, w, b.Bytes())
}

// NewVTTStage reads the reconciled "transcript" checkpoint and produces "vtt".
// It is not a speech model stage and performs no speaker/time inference.
func NewVTTStage() Stage {
	return Stage{Name: "vtt", Version: hash([]byte("speechjob-vtt-v3:transcript-schema2-source-timing:16k-floorstart-ceilend-namedrefs-literalquotes-generatedvoices-strictkeys")), Run: func(ctx context.Context, in *Input, w io.Writer) error {
		r, e := in.OpenCheckpoint(ctx, "transcript")
		if e != nil {
			return e
		}
		defer r.Close()
		t, e := ReadTranscriptJSON(ctx, r)
		if e != nil {
			return e
		}
		return WriteWebVTT(ctx, w, t)
	}}
}

// Require exact, complete root/cue keys. encoding/json otherwise accepts field
// aliases and treats missing/null speaker values as speaker 0. Escaped spellings
// are decoded before duplicate checks. The input is already byte-bounded.
func validateTranscriptJSONShape(ctx context.Context, b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var objectIndex int
	var value func(int) error
	value = func(depth int) error {
		if e := ctx.Err(); e != nil {
			return e
		}
		if depth > 8 {
			return fmt.Errorf("transcript JSON nesting exceeds bound")
		}
		t, e := d.Token()
		if e != nil {
			return e
		}
		delimiter, compound := t.(json.Delim)
		if !compound {
			if t == nil {
				return fmt.Errorf("null transcript value")
			}
			return nil
		}
		switch delimiter {
		case '{':
			index := objectIndex
			objectIndex++
			var required map[string]bool
			switch index {
			case 0:
				required = map[string]bool{"schema": true, "sample_rate": true, "total_samples": true, "language": true, "source_timing": true, "cues": true, "words": false}
			case 1:
				required = map[string]bool{"start_ns": true, "duration_ns": true, "exact": true, "has_edits": true, "source_rate": true, "priming": true, "padding": true, "leading_silence": true}
			default:
				required = map[string]bool{"start_sample": true, "end_sample": true, "speaker": true, "text": true}
			}
			seen := map[string]bool{}
			for d.More() {
				if e := ctx.Err(); e != nil {
					return e
				}
				key, e := d.Token()
				if e != nil {
					return e
				}
				name, ok := key.(string)
				_, allowed := required[name]
				if !ok || !allowed || seen[name] {
					return fmt.Errorf("duplicate/invalid transcript key %q in object %d", name, index)
				}
				seen[name] = true
				if e = value(depth + 1); e != nil {
					return e
				}
			}
			for name, mandatory := range required {
				if mandatory && !seen[name] {
					return fmt.Errorf("missing transcript key")
				}
			}
			end, e := d.Token()
			if e != nil || end != json.Delim('}') {
				return fmt.Errorf("unclosed transcript object")
			}
		case '[':
			for d.More() {
				if e = value(depth + 1); e != nil {
					return e
				}
			}
			end, e := d.Token()
			if e != nil || end != json.Delim(']') {
				return fmt.Errorf("unclosed transcript array")
			}
		default:
			return fmt.Errorf("unexpected transcript JSON delimiter")
		}
		return nil
	}
	if e := value(0); e != nil {
		return e
	}
	if _, e := d.Token(); e != io.EOF {
		return fmt.Errorf("trailing transcript JSON")
	}
	return ctx.Err()
}
