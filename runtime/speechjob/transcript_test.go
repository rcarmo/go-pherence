package speechjob

import (
	"bytes"
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func transcriptFixture() Transcript {
	return Transcript{Schema: 2, SampleRate: 16000, TotalSamples: 64000, Language: "pt", SourceTiming: media.SourceTiming{Start: 100 * time.Millisecond, Duration: 4 * time.Second, SourceRate: 48000}, Cues: []Cue{{StartSample: 1, EndSample: 17, Speaker: 0, Text: "Olá <script>& -->\n\nWEBVTT"}, {StartSample: 16000, EndSample: 32000, Speaker: -1, Text: "sem voz"}, {StartSample: 16000, EndSample: 64000, Speaker: 63, Text: "sobreposição"}}}
}
func TestTranscriptJSONAndVTTEscaping(t *testing.T) {
	ctx := context.Background()
	in := transcriptFixture()
	var b bytes.Buffer
	if e := WriteTranscriptJSON(ctx, &b, in); e != nil {
		t.Fatal(e)
	}
	got, e := ReadTranscriptJSON(ctx, bytes.NewReader(b.Bytes()))
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(got.Cues[0], in.Cues[0]) || got.SampleRate != 16000 || got.SourceTiming != in.SourceTiming {
		t.Fatal("round trip")
	}
	var vtt bytes.Buffer
	if e = WriteWebVTT(ctx, &vtt, got); e != nil {
		t.Fatal(e)
	}
	want := "WEBVTT\n\n1\n00:00:00.000 --> 00:00:00.002\n<v SPEAKER_00>Olá &lt;script&gt;&amp; --&gt;  WEBVTT</v>\n\n2\n00:00:01.000 --> 00:00:02.000\nsem voz\n\n3\n00:00:01.000 --> 00:00:04.000\n<v SPEAKER_63>sobreposição</v>\n\n"
	if vtt.String() != want {
		t.Fatalf("VTT\n%q\nwant%q", vtt.String(), want)
	}
	empty := Transcript{Schema: 2, SampleRate: 16000, Language: "fr"}
	b.Reset()
	if e = WriteTranscriptJSON(ctx, &b, empty); e != nil || !strings.Contains(b.String(), `"cues":[]`) {
		t.Fatal(b.String(), e)
	}
	vtt.Reset()
	if e = WriteWebVTT(ctx, &vtt, empty); e != nil || vtt.String() != "WEBVTT\n\n" {
		t.Fatal(e)
	}
	if vttTime(4*60*60*1000) != "04:00:00.000" {
		t.Fatal("hours")
	}
}
func TestTranscriptWordTimelineRoundTripAndValidation(t *testing.T) {
	tr := transcriptFixture()
	tr.Words = []WordCue{{StartSample: 200, EndSample: 400, Speaker: -1, Text: "Olá"}, {StartSample: 500, EndSample: 700, Speaker: 1, Text: "mundo"}}
	var out bytes.Buffer
	if err := WriteTranscriptJSON(context.Background(), &out, tr); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTranscriptJSON(context.Background(), bytes.NewReader(out.Bytes()))
	if err != nil || !reflect.DeepEqual(got.Words, tr.Words) {
		t.Fatal(out.String(), got.Words, err)
	}
	for _, mutate := range []func(*Transcript){
		func(t *Transcript) { t.Words[0].StartSample = -1 },
		func(t *Transcript) { t.Words[0].EndSample = t.Words[0].StartSample },
		func(t *Transcript) { t.Words[1].StartSample = 100 },
		func(t *Transcript) { t.Words[0].Speaker = 64 },
		func(t *Transcript) { t.Words[0].Text = " \n" },
	} {
		bad := tr
		bad.Words = append([]WordCue(nil), tr.Words...)
		mutate(&bad)
		if err := WriteTranscriptJSON(context.Background(), io.Discard, bad); err == nil {
			t.Fatal("invalid word timeline accepted", bad.Words)
		}
	}
}

func TestTranscriptRejectsBeforeWriting(t *testing.T) {
	for _, kind := range []string{"schema", "rate", "duration", "language", "start", "end", "extent", "ordering", "speaker", "empty", "control", "utf8", "size"} {
		tr := transcriptFixture()
		switch kind {
		case "schema":
			tr.Schema = 1
		case "rate":
			tr.SampleRate = 8000
		case "duration":
			tr.TotalSamples = 4*3600*16000 + 1
		case "language":
			tr.Language = "en\nX"
		case "start":
			tr.Cues[0].StartSample = -1
		case "end":
			tr.Cues[0].EndSample = 1
		case "extent":
			tr.Cues[2].EndSample++
		case "ordering":
			tr.Cues[2].StartSample = 0
		case "speaker":
			tr.Cues[0].Speaker = -2
		case "empty":
			tr.Cues[0].Text = " \n "
		case "control":
			tr.Cues[0].Text = "hi\x00"
		case "utf8":
			tr.Cues[0].Text = string([]byte{255})
		case "size":
			tr.Cues[0].Text = strings.Repeat("a", 65537)
		}
		for _, write := range []func(context.Context, io.Writer, Transcript) error{WriteTranscriptJSON, WriteWebVTT} {
			var out bytes.Buffer
			if e := write(context.Background(), &out, tr); e == nil || out.Len() != 0 {
				t.Fatal("accepted/wrote", kind, e)
			}
		}
	}
	for _, raw := range []string{`{"schema":1,"unknown":0}`, `{"schema":1,"schema":1}`, `{"schema":1,"\u0073chema":1}`, `{"cues":[{"text":"a","text":"b"}]}`, `{"schema":1} {}`, `null`, strings.Repeat("x", MaxTranscriptBytes+1)} {
		if _, e := ReadTranscriptJSON(context.Background(), strings.NewReader(raw)); e == nil {
			t.Fatal("invalid json")
		}
	}
}

type failingTranscriptWriter struct{ n int }

func (w *failingTranscriptWriter) Write(b []byte) (int, error) { w.n++; return 0, io.ErrClosedPipe }
func TestTranscriptCancellationAndWriterErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, write := range []func(context.Context, io.Writer, Transcript) error{WriteTranscriptJSON, WriteWebVTT} {
		w := &failingTranscriptWriter{}
		if e := write(ctx, w, transcriptFixture()); !errors.Is(e, context.Canceled) || w.n != 0 {
			t.Fatal(e)
		}
		if e := write(context.Background(), w, transcriptFixture()); !errors.Is(e, io.ErrClosedPipe) {
			t.Fatal(e)
		}
	}
	if _, e := ReadTranscriptJSON(ctx, strings.NewReader("")); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}
func TestVTTJobResumeAndLaterFailure(t *testing.T) {
	s, _ := openTest(t)
	m := createTest(t, s)
	transcriptRuns := 0
	tr := stage("transcript", func(ctx context.Context, _ *Input, w io.Writer) error {
		transcriptRuns++
		return WriteTranscriptJSON(ctx, w, transcriptFixture())
	})
	later := stage("speaker-json", func(context.Context, *Input, io.Writer) error { return io.ErrUnexpectedEOF })
	m, e := s.Run(context.Background(), m.ID, config, []Stage{tr, NewVTTStage(), later}, nil)
	if !errors.Is(e, io.ErrUnexpectedEOF) || len(m.Checkpoints) != 2 {
		t.Fatal(m, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), m.ID, "vtt")
	if !strings.Contains(readAll(t, r, e), "<v SPEAKER_00>") {
		t.Fatal("missing VTT")
	}
	later.Run = func(_ context.Context, _ *Input, w io.Writer) error { _, e := w.Write([]byte("{}")); return e }
	m, e = s.Run(context.Background(), m.ID, config, []Stage{tr, NewVTTStage(), later}, nil)
	if e != nil || m.Status != Complete || transcriptRuns != 1 {
		t.Fatal(m, e, transcriptRuns)
	}
}

func TestTranscriptJSONShape(t *testing.T) {
	ctx := context.Background()
	var encoded bytes.Buffer
	if e := WriteTranscriptJSON(ctx, &encoded, transcriptFixture()); e != nil {
		t.Fatal(e)
	}
	valid := encoded.String()
	for _, data := range []string{
		strings.Replace(valid, `"schema":2`, `"schema":2,"Schema":2`, 1),
		strings.Replace(valid, `"schema":2`, `"schema":2,"\u0073chema":2`, 1),
		strings.Replace(valid, `"schema":2`, `"Schema":2`, 1),
		strings.Replace(valid, `"source_timing":{`, `"source_timing":{"unknown":0,`, 1),
		strings.Replace(valid, `"source_timing":{`, `"source_timing":null,"ignored":{`, 1),
		strings.Replace(valid, `"speaker":0,`, ``, 1),
		strings.Replace(valid, `"speaker":0`, `"speaker":null`, 1),
		strings.Replace(valid, `"speaker":0`, `"speaker":0,"speaker":1`, 1),
		strings.Replace(valid, `Olá`, string([]byte{255}), 1),
		strings.Repeat("[", 10) + "0" + strings.Repeat("]", 10),
	} {
		if _, e := ReadTranscriptJSON(ctx, strings.NewReader(data)); e == nil {
			t.Fatal("accepted ambiguous/malformed JSON", data)
		}
	}
	// Keys may repeat across separate cues, but never within one object.
	if _, e := ReadTranscriptJSON(ctx, strings.NewReader(valid)); e != nil {
		t.Fatal(e)
	}
}

func TestWebVTTQuotesAreLiteral(t *testing.T) {
	tr := transcriptFixture()
	tr.Cues = tr.Cues[:1]
	tr.Cues[0].Text = `"d'água" & &#39; <v forged>`
	var out bytes.Buffer
	if e := WriteWebVTT(context.Background(), &out, tr); e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(out.String(), `"d'água" &amp; &amp;#39; &lt;v forged&gt;`) {
		t.Fatal(out.String())
	}
}
