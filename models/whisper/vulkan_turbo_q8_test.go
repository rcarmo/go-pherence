package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

func TestVulkanTurboQ8Robustness(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_Q8_ROBUSTNESS") != "1" {
		t.Skip("explicit Turbo Q8 silence/multi-window qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 3*time.Minute {
		t.Fatal("Turbo Q8 robustness qualification requires timeout<=3m")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 170*time.Second)
	defer cancel()
	model, tok, policy := pinnedTurboSpeechModel(t, ctx)
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
		after := vk.VulkanMemoryStats()
		if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
			t.Error("Turbo Q8 robustness leak", before, after)
		}
	}()
	jfk := os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH")
	pinnedSpeechFile(t, jfk, "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
	r, err := media.OpenCanonicalPCM(ctx, jfk)
	if err != nil {
		t.Fatal(err)
	}
	speech := make([]float32, 176000)
	n, readErr := r.ReadSamplesAt(ctx, speech, 0)
	closeErr := r.Close()
	if n != len(speech) || readErr != nil || closeErr != nil {
		t.Fatal("JFK read", n, readErr, closeErr)
	}
	type robustnessFixture struct {
		name, path, reference string
		samples               int64
		skip                  bool
	}
	dir := t.TempDir()
	makeFixture := func(name string, pcm []float32, reference string, skip bool) robustnessFixture {
		path := filepath.Join(dir, name+".wav")
		writeSpeechFixturePCM(t, path, pcm)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(data)
		t.Logf("TURBO_Q8_ROBUSTNESS_FIXTURE name=%s sha256=%s samples=%d skip=%t", name, hex.EncodeToString(hash[:]), len(pcm), skip)
		return robustnessFixture{name: name, path: path, reference: reference, samples: int64(len(pcm)), skip: skip}
	}
	const jfkText = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
	silence := make([]float32, 5*16000)
	long := make([]float32, 63*16000)
	copy(long[2*16000:], speech)
	copy(long[42*16000:], speech)
	fixtures := []robustnessFixture{
		makeFixture("silence-default-5s", silence, "", false),
		makeFixture("silence-skip-5s", silence, "", true),
		makeFixture("jfk-three-windows-63s", long, jfkText+" "+jfkText, true),
	}
	run := func(pathName string, enc *VulkanEncoder) (map[string][]WindowTranscript, int64) {
		outputs := map[string][]WindowTranscript{}
		start := time.Now()
		for _, fixture := range fixtures {
			reader, err := media.OpenCanonicalPCM(ctx, fixture.path)
			if err != nil {
				t.Fatal(err)
			}
			var windows []WindowTranscript
			err = model.TranscribePCMWindows(ctx, reader, fixture.samples, tok, PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96, SkipDigitalSilence: fixture.skip, VulkanEncoder: enc}, func(w WindowTranscript) error { windows = append(windows, w); return nil })
			closeErr := reader.Close()
			if err != nil || closeErr != nil {
				t.Fatal(fixture.name, err, closeErr)
			}
			outputs[fixture.name] = windows
		}
		ns := time.Since(start).Nanoseconds()
		t.Logf("TURBO_Q8_ROBUSTNESS_SAMPLE path=%s wall_ns=%d", pathName, ns)
		return outputs, ns
	}
	base, err := NewVulkanEncoder(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	baseline, baseNS := run("f32", base)
	if err := base.Close(); err != nil {
		t.Fatal(err)
	}
	q8, err := NewVulkanEncoderQ8Weight(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer q8.Close()
	fullStats := q8.Stats()
	quantized, q8NS := run("q8-all", q8)
	if err := q8.Close(); err != nil {
		t.Fatal(err)
	}
	mlp, err := NewVulkanEncoderQ8MLPWeight(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer mlp.Close()
	mlpStats := mlp.Stats()
	mlpQuantized, mlpNS := run("q8-mlp", mlp)
	if err := mlp.Close(); err != nil {
		t.Fatal(err)
	}
	kvmlp, err := NewVulkanEncoderQ8KVMLPWeight(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer kvmlp.Close()
	kvmlpStats := kvmlp.Stats()
	kvmlpQuantized, kvmlpNS := run("q8-kv-mlp", kvmlp)
	if err := kvmlp.Close(); err != nil {
		t.Fatal(err)
	}
	fullExact := true
	for _, fixture := range fixtures {
		bw, qw, mw, kw := baseline[fixture.name], quantized[fixture.name], mlpQuantized[fixture.name], kvmlpQuantized[fixture.name]
		q8Exact, mlpExact, kvmlpExact := reflect.DeepEqual(bw, qw), reflect.DeepEqual(bw, mw), reflect.DeepEqual(bw, kw)
		fullExact = fullExact && q8Exact
		if !mlpExact || !kvmlpExact {
			baselineJSON, _ := json.Marshal(bw)
			candidateJSON, _ := json.Marshal(kw)
			t.Fatalf("selective Q8 robustness transcript changed %s mlp=%t kvmlp=%t\nf32=%s\nq8_kv_mlp=%s", fixture.name, mlpExact, kvmlpExact, baselineJSON, candidateJSON)
		}
		wantWindows := int((fixture.samples + 479999) / 480000)
		if len(qw) != wantWindows || len(mw) != wantWindows || len(kw) != wantWindows {
			t.Fatal("robustness window count", fixture.name, len(qw), len(mw), len(kw), wantWindows)
		}
		text := func(windows []WindowTranscript) string {
			value := ""
			for _, w := range windows {
				for _, s := range w.Segments {
					value += " " + s.Text
				}
			}
			return strings.TrimSpace(value)
		}
		q8Text, mlpText, kvmlpText := text(qw), text(mw), text(kw)
		q8Edits, words := speechFixtureWER(fixture.reference, q8Text)
		mlpEdits, _ := speechFixtureWER(fixture.reference, mlpText)
		kvmlpEdits, _ := speechFixtureWER(fixture.reference, kvmlpText)
		if fixture.skip && fixture.reference == "" && (mlpText != "" || len(mw[0].Segments) != 0 || kvmlpText != "" || len(kw[0].Segments) != 0 || q8Text != "" || len(qw[0].Segments) != 0) {
			t.Fatal("silence skip emitted output")
		}
		if fixture.name == "jfk-three-windows-63s" && (q8Edits != 0 || mlpEdits != 0 || kvmlpEdits != 0 || words != 44 || len(qw[2].Segments) != 0 || len(mw[2].Segments) != 0 || len(kw[2].Segments) != 0) {
			t.Fatal("multi-window content regression", q8Edits, mlpEdits, kvmlpEdits, words)
		}
		entry, _ := json.Marshal(map[string]any{"fixture": fixture.name, "skip_digital_silence": fixture.skip, "windows": len(qw), "full_q8_exact_tokens_timestamps": q8Exact, "mlp_q8_exact_tokens_timestamps": mlpExact, "kv_mlp_q8_exact_tokens_timestamps": kvmlpExact, "full_q8_word_edits": q8Edits, "mlp_q8_word_edits": mlpEdits, "kv_mlp_q8_word_edits": kvmlpEdits, "reference_words": words, "full_q8_text": q8Text, "mlp_q8_text": mlpText, "kv_mlp_q8_text": kvmlpText})
		t.Log("TURBO_Q8_ROBUSTNESS " + string(entry))
	}
	result, _ := json.Marshal(map[string]any{"device": device, "fixtures": len(fixtures), "f32_ns": baseNS, "q8_all_ns": q8NS, "q8_mlp_ns": mlpNS, "q8_kv_mlp_ns": kvmlpNS, "q8_all_speedup": float64(baseNS) / float64(q8NS), "q8_mlp_speedup": float64(baseNS) / float64(mlpNS), "q8_kv_mlp_speedup": float64(baseNS) / float64(kvmlpNS), "full_q8_all_exact": fullExact, "mlp_q8_all_exact": true, "kv_mlp_q8_all_exact": true, "full_q8_stats": fullStats, "mlp_q8_stats": mlpStats, "kv_mlp_q8_stats": kvmlpStats})
	t.Log("TURBO_Q8_ROBUSTNESS_RESULT " + string(result))
}

func TestVulkanTurboQ8Podcast(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_Q8_PODCAST") != "1" {
		t.Skip("explicit Turbo selective-Q8 natural long-form qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 3*time.Minute {
		t.Fatal("Turbo selective-Q8 podcast qualification requires timeout<=3m")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 170*time.Second)
	defer cancel()
	model, tok, policy := pinnedTurboSpeechModel(t, ctx)
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
		after := vk.VulkanMemoryStats()
		if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
			t.Error("Turbo selective-Q8 podcast leak", before, after)
		}
	}()
	const sourceSHA = "8a7f5ea6b05a686ef1a6455d2a1683ed6cd1497a524efcb2cbfe10e810df6601"
	const startSample = int64(300 * 16000)
	const samples = int64(90 * 16000)
	podcast := os.Getenv("GO_PHERENCE_WHISPER_PODCAST_PATH")
	pinnedSpeechFile(t, podcast, sourceSHA)
	r, err := media.OpenCanonicalPCM(ctx, podcast)
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, samples)
	n, readErr := r.ReadSamplesAt(ctx, pcm, startSample)
	closeErr := r.Close()
	if n != len(pcm) || readErr != nil || closeErr != nil {
		t.Fatal("podcast window read", n, readErr, closeErr)
	}
	windowPath := filepath.Join(t.TempDir(), "podcast-300s-90s.wav")
	writeSpeechFixturePCM(t, windowPath, pcm)
	run := func(name string, enc *VulkanEncoder) ([]WindowTranscript, int64) {
		reader, err := media.OpenCanonicalPCM(ctx, windowPath)
		if err != nil {
			t.Fatal(err)
		}
		var windows []WindowTranscript
		start := time.Now()
		err = model.TranscribePCMWindows(ctx, reader, samples, tok, PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96, VulkanEncoder: enc}, func(w WindowTranscript) error {
			windows = append(windows, w)
			return nil
		})
		ns := time.Since(start).Nanoseconds()
		closeErr := reader.Close()
		if err != nil || closeErr != nil {
			t.Fatal(name, err, closeErr)
		}
		t.Logf("TURBO_Q8_PODCAST_SAMPLE path=%s wall_ns=%d", name, ns)
		return windows, ns
	}
	base, err := NewVulkanEncoder(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	baseline, baseNS := run("f32", base)
	if err = base.Close(); err != nil {
		t.Fatal(err)
	}
	q8, err := NewVulkanEncoderQ8KVMLPWeight(ctx, model.Encoder, 3000)
	if err != nil {
		t.Fatal(err)
	}
	defer q8.Close()
	quantized, q8NS := run("q8-kv-mlp", q8)
	stats := q8.Stats()
	if err = q8.Close(); err != nil {
		t.Fatal(err)
	}
	if len(baseline) != 3 || len(quantized) != 3 {
		t.Fatal("incomplete podcast windows", len(baseline), len(quantized))
	}
	if !reflect.DeepEqual(baseline, quantized) {
		baselineJSON, _ := json.Marshal(baseline)
		candidateJSON, _ := json.Marshal(quantized)
		t.Fatalf("selective Q8 podcast transcript changed\nf32=%s\nq8_kv_mlp=%s", baselineJSON, candidateJSON)
	}
	segments, tokens := 0, 0
	for _, window := range quantized {
		segments += len(window.Segments)
		for _, segment := range window.Segments {
			tokens += len(segment.Tokens)
		}
	}
	if segments == 0 || tokens == 0 {
		t.Fatal("podcast gate produced no speech", segments, tokens)
	}
	result, _ := json.Marshal(map[string]any{"device": device, "source_sha256": sourceSHA, "start_samples": startSample, "samples": samples, "windows": len(quantized), "segments": segments, "tokens": tokens, "exact_tokens_timestamps": true, "f32_ns": baseNS, "q8_kv_mlp_ns": q8NS, "speedup": float64(baseNS) / float64(q8NS), "q8_kv_mlp_stats": stats, "labeled_quality": false})
	t.Log("TURBO_Q8_PODCAST_RESULT " + string(result))
}

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
