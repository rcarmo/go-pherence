package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func TestVulkanTurboQ8Weight(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_Q8_WEIGHT") != "1" {
		t.Skip("explicit trained Turbo Q8-weight qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 3*time.Minute {
		t.Fatal("Turbo Q8-weight qualification requires timeout<=3m")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 170*time.Second)
	defer cancel()
	model, tok, policy := pinnedTurboSpeechModel(t, ctx)
	dequant, qbytes, fbytes := q8ProjectionEncoder(model.Encoder)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	wantDevice := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	lower := strings.ToLower(device)
	if wantDevice == "" || !strings.Contains(device, wantDevice) || strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("unexpected physical device", device)
	}
	before := vk.VulkanMemoryStats()
	if before.Allocations != 0 || before.Bytes != 0 {
		t.Fatal("isolated process required", before)
	}
	if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 4 << 30, MaxAllocations: 40}); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	}()
	defer func() {
		after := vk.VulkanMemoryStats()
		if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
			t.Error("Turbo Q8 leak", before, after)
		}
	}()
	const frames = 4
	mel := make([]float32, model.Config.NumMelBins*frames)
	for i := range mel {
		mel[i] = float32(i%17-8) / 32
	}
	base, e := NewVulkanEncoder(ctx, model.Encoder, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer base.Close()
	baseStats := base.Stats()
	bo, e := base.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	if e = base.Close(); e != nil {
		t.Fatal(e)
	}
	deq, e := NewVulkanEncoder(ctx, dequant, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer deq.Close()
	do, e := deq.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	if e = deq.Close(); e != nil {
		t.Fatal(e)
	}
	q8, e := NewVulkanEncoderQ8Weight(ctx, model.Encoder, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer q8.Close()
	q8Stats := q8.Stats()
	qo, e := q8.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	metrics := func(a, b []float32) (float64, float64, float64) {
		mx, rms, ref := 0., 0., 0.
		for i, v := range a {
			d := float64(v - b[i])
			mx = math.Max(mx, math.Abs(d))
			rms += d * d
			ref += float64(b[i]) * float64(b[i])
		}
		return mx, math.Sqrt(rms / float64(len(a))), math.Sqrt(ref / float64(len(a)))
	}
	kmx, krms, kref := metrics(qo, do)
	qmx, qrms, qref := metrics(qo, bo)
	bs, e := NewDecoderStateContext(ctx, model.Config, bo, frames/2, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	qs, e := NewDecoderStateContext(ctx, model.Config, qo, frames/2, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	top := func(a []float32) int {
		j := 0
		for i, v := range a {
			if v > a[j] {
				j = i
			}
		}
		return j
	}
	steps := []map[string]any{}
	for i, token := range []int{50258, 50259, 50359} {
		b := append([]float32(nil), model.Decoder.ForwardToken(token, bs)...)
		q := model.Decoder.ForwardToken(token, qs)
		bm, qm := top(b), top(q)
		md := 0.
		for j, v := range q {
			md = math.Max(md, math.Abs(float64(v-b[j])))
		}
		steps = append(steps, map[string]any{"step": i, "baseline_top": bm, "q8_top": qm, "max_logit_abs": md})
		if bm != qm {
			t.Fatal("top1", i, bm, qm)
		}
	}
	record, _ := json.Marshal(map[string]any{"values": len(qo), "packed_vs_dequant_max_abs": kmx, "packed_vs_dequant_rms": krms, "dequant_reference_rms": kref, "q8_vs_f32_max_abs": qmx, "q8_vs_f32_rms": qrms, "f32_reference_rms": qref, "q8_rms_ratio": qrms / qref, "f32_projection_bytes": fbytes, "q8_projection_bytes": qbytes, "baseline_stats": baseStats, "q8_stats": q8Stats, "decoder_steps": steps, "memory": vk.VulkanMemoryStats()})
	t.Log("TURBO_Q8_SHORT " + string(record))
	if e = q8.Close(); e != nil {
		t.Fatal(e)
	}
	q8 = nil
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_Q8_SPEECH") != "1" {
		return
	}
	input := os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH")
	pinnedSpeechFile(t, input, "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
	run := func(name string, enc *VulkanEncoder) ([]WindowTranscript, int64) {
		r, err := media.OpenCanonicalPCM(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		var windows []WindowTranscript
		start := time.Now()
		err = model.TranscribePCMWindows(ctx, r, 176000, tok, PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96, VulkanEncoder: enc}, func(w WindowTranscript) error { windows = append(windows, w); return nil })
		ns := time.Since(start).Nanoseconds()
		closeErr := r.Close()
		if err != nil || closeErr != nil {
			t.Fatal(err, closeErr)
		}
		t.Logf("TURBO_Q8_SAMPLE path=%s wall_ns=%d", name, ns)
		return windows, ns
	}
	base, e = NewVulkanEncoder(ctx, model.Encoder, 3000)
	if e != nil {
		t.Fatal(e)
	}
	defer base.Close()
	bw, bns := run("f32", base)
	if e = base.Close(); e != nil {
		t.Fatal(e)
	}
	q8, e = NewVulkanEncoderQ8Weight(ctx, model.Encoder, 3000)
	if e != nil {
		t.Fatal(e)
	}
	defer q8.Close()
	qw, qns := run("q8", q8)
	if e = q8.Close(); e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(bw, qw) {
		t.Fatalf("Turbo Q8 transcript changed\nf32=%+v\nq8=%+v", bw, qw)
	}
	text := ""
	for _, w := range qw {
		for _, s := range w.Segments {
			text += " " + s.Text
		}
	}
	const reference = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
	edits, words := speechFixtureWER(reference, strings.TrimSpace(text))
	if edits != 0 || words != 22 {
		t.Fatal("Turbo Q8 JFK WER", edits, words, text)
	}
	result, _ := json.Marshal(map[string]any{"device": device, "exact_tokens_timestamps": true, "word_edits": edits, "reference_words": words, "f32_ns": bns, "q8_ns": qns, "speedup": float64(bns) / float64(qns), "text": strings.TrimSpace(text)})
	t.Log("TURBO_Q8_JFK " + string(result))
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_Q8_MULTILINGUAL") != "1" {
		return
	}
	root := os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR")
	if root == "" {
		t.Fatal("MINDS fixture directory required")
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := media.NewFFmpeg(media.Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	type decodedFixture struct {
		fixture publicSpeechFixture
		path    string
	}
	decoded := make([]decodedFixture, 0, len(mindsSpeechFixtures))
	for i, fixture := range mindsSpeechFixtures {
		fixture.File = filepath.Join(root, fixture.File)
		pinnedSpeechFile(t, fixture.File, fixture.SHA256)
		result, err := adapter.DecodeToFile(ctx, fixture.File, filepath.Join(t.TempDir(), fmt.Sprintf("minds-%d.wav", i)))
		if err != nil || int64(result.Timeline.Samples) != fixture.Samples || result.Timeline.SampleRate != 16000 {
			t.Fatal("MINDS decode", fixture.Name, result.Timeline, err)
		}
		decoded = append(decoded, decodedFixture{fixture: fixture, path: result.Path})
	}
	runCorpus := func(name string, enc *VulkanEncoder) (map[string][]WindowTranscript, int64) {
		outputs := map[string][]WindowTranscript{}
		start := time.Now()
		for _, item := range decoded {
			r, err := media.OpenCanonicalPCM(ctx, item.path)
			if err != nil {
				t.Fatal(err)
			}
			var windows []WindowTranscript
			err = model.TranscribePCMWindows(ctx, r, item.fixture.Samples, tok, PCMTranscribeOptions{Language: item.fixture.Language, Generation: policy, MaxNewTokens: 96, VulkanEncoder: enc}, func(w WindowTranscript) error { windows = append(windows, w); return nil })
			closeErr := r.Close()
			if err != nil || closeErr != nil {
				t.Fatal(item.fixture.Name, err, closeErr)
			}
			outputs[item.fixture.Name] = windows
		}
		ns := time.Since(start).Nanoseconds()
		t.Logf("TURBO_Q8_MULTILINGUAL_SAMPLE path=%s wall_ns=%d", name, ns)
		return outputs, ns
	}
	base, e = NewVulkanEncoder(ctx, model.Encoder, 3000)
	if e != nil {
		t.Fatal(e)
	}
	defer base.Close()
	baselineCorpus, baseNS := runCorpus("f32", base)
	if e = base.Close(); e != nil {
		t.Fatal(e)
	}
	q8, e = NewVulkanEncoderQ8Weight(ctx, model.Encoder, 3000)
	if e != nil {
		t.Fatal(e)
	}
	defer q8.Close()
	q8Corpus, q8NS := runCorpus("q8", q8)
	if e = q8.Close(); e != nil {
		t.Fatal(e)
	}
	for _, item := range decoded {
		bw, qw := baselineCorpus[item.fixture.Name], q8Corpus[item.fixture.Name]
		if !reflect.DeepEqual(bw, qw) {
			t.Fatal("multilingual transcript changed", item.fixture.Name)
		}
		text := ""
		for _, w := range qw {
			for _, s := range w.Segments {
				text += " " + s.Text
			}
		}
		edits, words := speechFixtureWER(item.fixture.Reference, strings.TrimSpace(text))
		if edits != 0 {
			t.Fatal("Turbo multilingual WER", item.fixture.Name, edits, words, text)
		}
		entry, _ := json.Marshal(map[string]any{"fixture": item.fixture.Name, "language": item.fixture.Language, "exact_tokens_timestamps": true, "word_edits": edits, "reference_words": words, "text": strings.TrimSpace(text)})
		t.Log("TURBO_Q8_MULTILINGUAL " + string(entry))
	}
	corpusResult, _ := json.Marshal(map[string]any{"fixtures": len(decoded), "f32_ns": baseNS, "q8_ns": q8NS, "speedup": float64(baseNS) / float64(q8NS), "all_exact": true, "all_zero_wer": true})
	t.Log("TURBO_Q8_MULTILINGUAL_RESULT " + string(corpusResult))
}
