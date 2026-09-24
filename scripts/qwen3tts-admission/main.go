//go:build linux

// Run from the repository root with the approved, pinned CustomVoice checkpoint:
// GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 go run ./scripts/qwen3tts-admission <checkpoint-dir>
// Append --mixed to test Hi/seed-42 against Hello world/seed-7 at 16 frames.
// Append --retain-3 for three rounds of two 32-frame Hi requests, retaining
// and rechecking earlier outputs while later rounds run.
// For ownership/race evidence, rerun with go run -race. Race instrumentation
// changes memory use: only the ordinary run is used for RSS admission.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

func verify(path, want string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	if size > 0 && stat.Size() != size {
		return fmt.Errorf("size mismatch: %s", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if fmt.Sprintf("%x", h.Sum(nil)) != want {
		return fmt.Errorf("hash mismatch: %s", path)
	}
	return nil
}

func inspect(label string) (heap uint64, highWaterKiB uint64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	b, e := os.ReadFile("/proc/self/status")
	if e != nil {
		panic(e)
	}
	var rss, hwm string
	for _, line := range splitLines(string(b)) {
		if len(line) > 6 && line[:6] == "VmRSS:" {
			rss = line
		}
		if len(line) > 6 && line[:6] == "VmHWM:" {
			hwm = line
		}
	}
	fmt.Printf("%s heap_alloc=%d heap_inuse=%d heap_sys=%d rss=%q hwm=%q\n", label, m.HeapAlloc, m.HeapInuse, m.HeapSys, rss, hwm)
	if _, err := fmt.Sscanf(hwm, "VmHWM: %d kB", &highWaterKiB); err != nil {
		panic(err)
	}
	return m.HeapAlloc, highWaterKiB
}
func splitLines(s string) []string {
	out := make([]string, 0, 50)
	begin := 0
	for i := range s {
		if s[i] == '\n' {
			out = append(out, s[begin:i])
			begin = i + 1
		}
	}
	return out
}
func check(r qwen3tts.BoundedCPUResult, codes, wave []byte, frames int, threshold float64) float64 {
	if len(r.Semantic) != frames || len(r.Acoustic) != frames*15 || len(r.Waveform) != frames*1920 {
		panic(fmt.Sprintf("geometry %d/%d/%d", len(r.Semantic), len(r.Acoustic), len(r.Waveform)))
	}
	for f := 0; f < frames; f++ {
		if r.Semantic[f] != binary.LittleEndian.Uint32(codes[f*64:]) || r.Semantic[f] == qwen3tts.CodecEOS {
			panic(fmt.Sprintf("semantic frame %d", f))
		}
		for g := 0; g < 15; g++ {
			if r.Acoustic[f*15+g] != binary.LittleEndian.Uint32(codes[f*64+(g+1)*4:]) {
				panic(fmt.Sprintf("acoustic frame %d group %d", f, g))
			}
		}
	}
	var max float64
	for i, v := range r.Waveform {
		want := math.Float32frombits(binary.LittleEndian.Uint32(wave[4*i:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			panic("nonfinite waveform")
		}
		diff := math.Abs(float64(v) - float64(want))
		if diff > max {
			max = diff
		}
	}
	if max > threshold {
		panic(fmt.Sprintf("wave max %g > %g", max, threshold))
	}
	return max
}
func main() {
	if len(os.Args) != 2 && (len(os.Args) != 3 || (os.Args[2] != "--mixed" && os.Args[2] != "--retain-3")) {
		panic("usage: probe <pinned-checkpoint-dir> [--mixed|--retain-3]")
	}
	dir := os.Args[1]
	mixed := len(os.Args) == 3 && os.Args[2] == "--mixed"
	retain := len(os.Args) == 3 && os.Args[2] == "--retain-3"
	for _, entry := range []struct {
		path, sha string
		size      int64
	}{
		{filepath.Join(dir, "model.safetensors"), "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb", 1811626576},
		{filepath.Join(dir, "config.json"), "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455", 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258", 682293092},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167", 0},
	} {
		if e := verify(entry.path, entry.sha, entry.size); e != nil {
			panic(e)
		}
	}
	root := "model/qwen3tts/testdata/customvoice_0b6_ryan_hello"
	data, e := os.ReadFile(filepath.Join(root, "reference.json"))
	if e != nil {
		panic(e)
	}
	var ref struct {
		Oracle         string  `json:"oracle_revision"`
		Threshold32    float64 `json:"hi_32_go_waveform_max_abs_threshold"`
		ThresholdHi    float64 `json:"hi_seeded_sixteen_waveform_max_abs_threshold"`
		ThresholdSeed7 float64 `json:"seed7_sixteen_waveform_max_abs_threshold"`
	}
	if e = json.Unmarshal(data, &ref); e != nil {
		panic(e)
	}
	if ref.Oracle != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.Threshold32 != 1.6e-6 || ref.ThresholdHi != 1.6e-6 || ref.ThresholdSeed7 != 3.5e-6 {
		panic("reference pin changed")
	}
	type fixture struct {
		text        string
		seed        uint64
		frames      int
		codes, wave []byte
		threshold   float64
	}
	fixtures := [2]fixture{}
	files := []struct {
		codeName, codeSHA, waveName, waveSHA, script, scriptSHA, text string
		seed                                                          uint64
		frames                                                        int
		threshold                                                     float64
	}{
		{"probe_hi_32_codes.u32le", "88aafb544a81f0f09d4b83125c9b4f01370ade24fe24f47e9fcde86818a58391", "probe_hi_32_waveform.f32le", "831831b02d90b880311275807008585f82a9425f32db7401a4c6996397039a82", "qwen3tts_probe_hi_32_eos.rs", "8acce5b9e8fb6cae755f23ae019416329a26b9fad1529f773bd0043e5e1b91b6", "Hi", 42, 32, ref.Threshold32},
	}
	if mixed {
		files = []struct {
			codeName, codeSHA, waveName, waveSHA, script, scriptSHA, text string
			seed                                                          uint64
			frames                                                        int
			threshold                                                     float64
		}{
			{"hi_seeded_sixteen_codes.u32le", "17b16ec184d5a075805c057aa5eaa91b1960f97710f0f06206c6676f8b861a52", "hi_seeded_sixteen_waveform.f32le", "373f8c3f6f0f9b2e39a81487584160fbf183813001ea303dfb0869c816df740f", "qwen3tts_oracle_hi_seeded_sixteen.rs", "a28aac5b628164f01176773b753a1bbbfbf5efce56d9c01b2fa1e1666fb002ee", "Hi", 42, 16, ref.ThresholdHi},
			{"seed7_sixteen_codes.u32le", "3d955a9dd5074d8be14976d0c53afbb0704b52f58ade0584b8bfe740abc9a92d", "seed7_sixteen_waveform.f32le", "01de12600f88be6964a76bdbe0e99eed925c4b4b1aa75444a1bd934a83ff9a06", "qwen3tts_oracle_seed7_sixteen_frames.rs", "cf9639ee01098a07516d9f6599abec6154fa3cd9c213e58a4bfc44dfdaa46406", "Hello world", 7, 16, ref.ThresholdSeed7},
		}
	}
	if !mixed {
		files = append(files, files[0])
	}
	for i, f := range files {
		for _, check := range []struct {
			path, hash string
			size       int64
		}{
			{filepath.Join("scripts", f.script), f.scriptSHA, 0},
			{filepath.Join(root, f.codeName), f.codeSHA, int64(f.frames * 16 * 4)},
			{filepath.Join(root, f.waveName), f.waveSHA, int64(f.frames * 1920 * 4)},
		} {
			if e := verify(check.path, check.hash, check.size); e != nil {
				panic(e)
			}
		}
		codes, e := os.ReadFile(filepath.Join(root, f.codeName))
		if e != nil {
			panic(e)
		}
		wave, e := os.ReadFile(filepath.Join(root, f.waveName))
		if e != nil {
			panic(e)
		}
		fixtures[i] = fixture{f.text, f.seed, f.frames, codes, wave, f.threshold}
	}
	cfg, e := qwen3tts.ReadModelDir(dir)
	if e != nil {
		panic(e)
	}
	t, e := qwen3tts.LoadTalkerCPUFromDir(dir, cfg)
	if e != nil {
		panic(e)
	}
	p, e := qwen3tts.LoadCodePredictorCPUFromDir(dir, cfg)
	if e != nil {
		panic(e)
	}
	d, e := qwen3tts.LoadDecoder12HzCPUFromDir(dir)
	if e != nil {
		panic(e)
	}
	tok, e := qwen3tts.LoadTokenizer(dir)
	if e != nil {
		panic(e)
	}
	var plans [2]qwen3tts.RuntimeRequestPlan
	for i, f := range fixtures {
		prompt, e := qwen3tts.BuildCustomVoicePrompt(tok, f.text, qwen3tts.Ryan, qwen3tts.English)
		if e != nil {
			panic(e)
		}
		if f.text == "Hi" && (len(prompt.Text) != 10 || prompt.Text[9] != 13048) {
			panic("Hi token changed")
		}
		plans[i], e = qwen3tts.NewRuntimeRequestPlan(cfg, qwen3tts.RuntimeRequest{Conditioning: qwen3tts.ConditioningRequest{Speaker: qwen3tts.Ryan, Language: qwen3tts.English}, Prompt: prompt, MaxFrames: f.frames})
		if e != nil {
			panic(e)
		}
	}
	runtime.GC()
	base, _ := inspect("before_concurrent_post_gc")
	if retain {
		// Scope the retained results in a call, so the final GC observes
		// released outputs, not compiler-dependent liveness of loop locals.
		func() {
			var held [][2]qwen3tts.BoundedCPUResult
			for round := 0; round < 3; round++ {
				var results [2]qwen3tts.BoundedCPUResult
				var errs [2]error
				var wg sync.WaitGroup
				start := make(chan struct{})
				for i := range results {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						<-start
						results[i], errs[i] = qwen3tts.GenerateCappedSeededCPU(plans[i], t, p, d, fixtures[i].seed)
					}(i)
				}
				begin := time.Now()
				close(start)
				wg.Wait()
				fmt.Printf("round=%d wall=%s\n", round, time.Since(begin))
				for i, r := range results {
					if errs[i] != nil {
						panic(errs[i])
					}
					f := fixtures[i]
					fmt.Printf("round=%d request_%d max_abs=%g\n", round, i, check(r, f.codes, f.wave, f.frames, f.threshold))
				}
				held = append(held, results)
				for a, pair := range held {
					for i, r := range pair {
						f := fixtures[i]
						check(r, f.codes, f.wave, f.frames, f.threshold)
						for b, earlier := range held[:a+1] {
							for j, prior := range earlier {
								if a == b && i == j {
									continue
								}
								if &r.Semantic[0] == &prior.Semantic[0] || &r.Acoustic[0] == &prior.Acoustic[0] || &r.Waveform[0] == &prior.Waveform[0] {
									panic("retained outputs alias")
								}
							}
						}
					}
				}
				runtime.GC()
				inspect(fmt.Sprintf("round_%d_retained_post_gc", round))
				runtime.KeepAlive(held)
			}
		}()
		runtime.GC()
		after, peak := inspect("after_release_post_gc")
		if after > base+1048576 {
			panic(fmt.Sprintf("retained heap grew by %d bytes", after-base))
		}
		runtime.KeepAlive(t)
		runtime.KeepAlive(p)
		runtime.KeepAlive(d)
		runtime.KeepAlive(tok)
		runtime.KeepAlive(plans)
		fmt.Printf("post_gc_delta_bytes=%d peak_rss_kib=%d\n", int64(after)-int64(base), peak)
		return
	}
	var results [2]qwen3tts.BoundedCPUResult
	var errs [2]error
	var took [2]time.Duration
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			begin := time.Now()
			results[i], errs[i] = qwen3tts.GenerateCappedSeededCPU(plans[i], t, p, d, fixtures[i].seed)
			took[i] = time.Since(begin)
		}(i)
	}
	begin := time.Now()
	close(start)
	wg.Wait()
	fmt.Printf("both_returned wall=%s request0=%s request1=%s\n", time.Since(begin), took[0], took[1])
	inspect("both_returned")
	for i, r := range results {
		if errs[i] != nil {
			panic(errs[i])
		}
		f := fixtures[i]
		fmt.Printf("request_%d text=%q seed=%d max_abs=%g\n", i, f.text, f.seed, check(r, f.codes, f.wave, f.frames, f.threshold))
	}
	if &results[0].Semantic[0] == &results[1].Semantic[0] || &results[0].Acoustic[0] == &results[1].Acoustic[0] || &results[0].Waveform[0] == &results[1].Waveform[0] {
		panic("results alias")
	}
	orig := results[1].Waveform[0]
	results[0].Waveform[0]++
	if orig != results[1].Waveform[0] {
		panic("mutated peer result")
	}
	fmt.Println("independent_owned_results=true")
	results = [2]qwen3tts.BoundedCPUResult{}
	runtime.GC()
	after, finalPeak := inspect("after_concurrent_post_gc")
	if after > base+1048576 {
		panic(fmt.Sprintf("retained heap grew by %d bytes", after-base))
	}
	runtime.KeepAlive(t)
	runtime.KeepAlive(p)
	runtime.KeepAlive(d)
	runtime.KeepAlive(tok)
	runtime.KeepAlive(plans)
	fmt.Printf("post_gc_delta_bytes=%d peak_rss_kib=%d\n", int64(after)-int64(base), finalPeak)
	// Race instrumentation adds shadow memory, so apply the 6 GiB RSS
	// admission threshold only to the ordinary (non-race) run log.
}
