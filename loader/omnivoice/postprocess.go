package omnivoice

import (
	"fmt"
	"math"
)

const (
	Mono24kSampleRate     = 24000
	maxReferenceSamples24 = 20 * Mono24kSampleRate
	maxOutputSamples24    = 10 * Mono24kSampleRate
)

// SilenceOptions mirrors omnivoice.utils.audio.remove_silence for raw mono
// float32 24 kHz audio. Processing matches upstream order: float32 -> PCM16
// quantization, pydub-style silence detection/splitting, edge trimming, then
// PCM16 -> float32 conversion.
type SilenceOptions struct {
	MidSilenceMS       int
	LeadingKeepMS      int
	TrailingKeepMS     int
	SilenceThresholdDB float64
	MaxSamples         int
}

// DefaultSilenceOptions matches upstream remove_silence defaults.
func DefaultSilenceOptions() SilenceOptions {
	return SilenceOptions{
		MidSilenceMS:       300,
		LeadingKeepMS:      100,
		TrailingKeepMS:     300,
		SilenceThresholdDB: -50,
		MaxSamples:         maxReferenceSamples24,
	}
}

// ReferenceSilenceOptions matches the tighter trimming used for cached
// reference audio preparation.
func ReferenceSilenceOptions() SilenceOptions {
	return SilenceOptions{
		MidSilenceMS:       200,
		LeadingKeepMS:      100,
		TrailingKeepMS:     200,
		SilenceThresholdDB: -50,
		MaxSamples:         maxReferenceSamples24,
	}
}

// FadePadOptions mirrors omnivoice.utils.audio.fade_and_pad_audio for raw mono
// float32 24 kHz audio and adds independent fade flags.
type FadePadOptions struct {
	PadDuration  float64
	FadeDuration float64
	FadeIn       bool
	FadeOut      bool
	MaxSamples   int
}

// DefaultFadePadOptions matches upstream fade_and_pad_audio defaults and caps
// input at 10 seconds before padding. Returned audio is capped at 20 seconds.
func DefaultFadePadOptions() FadePadOptions {
	return FadePadOptions{
		PadDuration:  0.1,
		FadeDuration: 0.1,
		FadeIn:       true,
		FadeOut:      true,
		MaxSamples:   maxOutputSamples24,
	}
}

// RemoveSilenceMono24k removes middle silences and trims edge silences using
// upstream-compatible pydub semantics on quantized PCM16 windows.
func RemoveSilenceMono24k(audio []float32, opts SilenceOptions) ([]float32, error) {
	if opts.MidSilenceMS < 0 || opts.MidSilenceMS > 20000 || opts.LeadingKeepMS < 0 || opts.LeadingKeepMS > 20000 || opts.TrailingKeepMS < 0 || opts.TrailingKeepMS > 20000 {
		return nil, fmt.Errorf("omnivoice: silence durations must be nonnegative")
	}
	if math.IsNaN(opts.SilenceThresholdDB) || math.IsInf(opts.SilenceThresholdDB, 0) {
		return nil, fmt.Errorf("omnivoice: silence threshold must be finite")
	}
	if err := validateMono24kAudio(audio, opts.MaxSamples); err != nil {
		return nil, err
	}
	if len(audio) == 0 {
		return nil, nil
	}
	pcm := quantizeMonoPCM16(audio)
	if opts.MidSilenceMS > 0 {
		pcm = splitOnSilencePCM(pcm, opts.MidSilenceMS, opts.SilenceThresholdDB, opts.MidSilenceMS, 10)
	}
	pcm = removeSilenceEdgesPCM(pcm, opts.LeadingKeepMS, opts.TrailingKeepMS, opts.SilenceThresholdDB)
	return dequantizeMonoPCM16(pcm), nil
}

// RemoveReferenceSilenceMono24k applies the project reference preset
// (mid=200 ms, lead=100 ms, trail=200 ms, threshold=-50 dBFS).
func RemoveReferenceSilenceMono24k(audio []float32) ([]float32, error) {
	return RemoveSilenceMono24k(audio, ReferenceSilenceOptions())
}

// FadeAndPadMono24k applies upstream-compatible fades/padding to mono 24 kHz
// float32 audio. Fades are applied before padding.
func FadeAndPadMono24k(audio []float32, opts FadePadOptions) ([]float32, error) {
	if opts.PadDuration < 0 || opts.PadDuration > 10 || opts.FadeDuration < 0 || opts.FadeDuration > 20 || math.IsNaN(opts.PadDuration) || math.IsNaN(opts.FadeDuration) || math.IsInf(opts.PadDuration, 0) || math.IsInf(opts.FadeDuration, 0) {
		return nil, fmt.Errorf("omnivoice: fade/pad durations must be finite and nonnegative")
	}
	if err := validateMono24kAudio(audio, opts.MaxSamples); err != nil {
		return nil, err
	}
	if len(audio) == 0 {
		return nil, nil
	}
	fadeSamples := int(opts.FadeDuration * Mono24kSampleRate)
	padSamples := int(opts.PadDuration * Mono24kSampleRate)
	if len(audio)+2*padSamples > maxReferenceSamples24 {
		return nil, fmt.Errorf("omnivoice: padded output exceeds 20 seconds")
	}
	processed := append([]float32(nil), audio...)
	if fadeSamples > 0 {
		k := min(fadeSamples, len(processed)/2)
		if k > 0 {
			if opts.FadeIn {
				for i := 0; i < k; i++ {
					processed[i] *= linspaceValue(0, 1, i, k)
				}
			}
			if opts.FadeOut {
				start := len(processed) - k
				for i := 0; i < k; i++ {
					processed[start+i] *= linspaceValue(1, 0, i, k)
				}
			}
		}
	}
	if padSamples > 0 {
		out := make([]float32, len(processed)+2*padSamples)
		copy(out[padSamples:], processed)
		processed = out
	}
	return processed, nil
}

func validateMono24kAudio(audio []float32, maxSamples int) error {
	if maxSamples <= 0 || maxSamples > maxReferenceSamples24 {
		return fmt.Errorf("omnivoice: MaxSamples must be 1..480000")
	}
	if len(audio) > maxSamples {
		return fmt.Errorf("omnivoice: audio exceeds %d samples at 24 kHz", maxSamples)
	}
	for _, v := range audio {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("omnivoice: nonfinite audio")
		}
	}
	return nil
}

func quantizeMonoPCM16(audio []float32) []int16 {
	pcm := make([]int16, len(audio))
	for i, v := range audio {
		scaled := float64(v) * 32768.0
		switch {
		case math.IsNaN(scaled):
			pcm[i] = 0
		case math.IsInf(scaled, 1) || scaled > 32767:
			pcm[i] = 32767
		case math.IsInf(scaled, -1) || scaled < -32768:
			pcm[i] = -32768
		default:
			pcm[i] = int16(scaled)
		}
	}
	return pcm
}

func dequantizeMonoPCM16(pcm []int16) []float32 {
	out := make([]float32, len(pcm))
	for i, v := range pcm {
		out[i] = float32(v) / 32768.0
	}
	return out
}

func splitOnSilencePCM(pcm []int16, minSilenceMS int, silenceThresholdDB float64, keepSilenceMS int, seekStepMS int) []int16 {
	nonsilent := detectNonsilentPCM(pcm, minSilenceMS, silenceThresholdDB, seekStepMS)
	if len(nonsilent) == 0 {
		return nil
	}
	ranges := make([][2]int, len(nonsilent))
	for i, r := range nonsilent {
		ranges[i] = [2]int{r[0] - keepSilenceMS, r[1] + keepSilenceMS}
	}
	for i := 0; i+1 < len(ranges); i++ {
		lastEnd := ranges[i][1]
		nextStart := ranges[i+1][0]
		if nextStart < lastEnd {
			mid := (lastEnd + nextStart) / 2
			ranges[i][1] = mid
			ranges[i+1][0] = mid
		}
	}
	total := 0
	for _, r := range ranges {
		total += slicePCMExpectedFrames(pcm, r[0], r[1])
	}
	out := make([]int16, total)
	at := 0
	for _, r := range ranges {
		at += copyPCMSlice(out[at:], pcm, r[0], r[1])
	}
	return out
}

func removeSilenceEdgesPCM(pcm []int16, leadSilMS int, trailSilMS int, silenceThresholdDB float64) []int16 {
	start := detectLeadingSilencePCM(pcm, silenceThresholdDB, 10)
	start = max(0, start-leadSilMS)
	pcm = slicePCMByMS(pcm, start, audioLenMS(len(pcm)))
	if len(pcm) == 0 {
		return nil
	}
	rev := reversePCM(pcm)
	start = detectLeadingSilencePCM(rev, silenceThresholdDB, 10)
	start = max(0, start-trailSilMS)
	rev = slicePCMByMS(rev, start, audioLenMS(len(rev)))
	return reversePCM(rev)
}

func detectLeadingSilencePCM(pcm []int16, silenceThresholdDB float64, chunkMS int) int {
	if chunkMS <= 0 {
		panic("omnivoice: chunkMS must be positive")
	}
	prefix := squaredPrefixPCM(pcm)
	limit := rmsLimitForStrictLess(dbToAmplitude(silenceThresholdDB))
	trimMS := 0
	segLenMS := audioLenMS(len(pcm))
	chunkFrames := msToFrames(chunkMS)
	for trimMS < segLenMS {
		startFrame := msToFrames(trimMS)
		sum := windowSquaredSum(prefix, startFrame, chunkFrames, len(pcm))
		if !rmsAtMost(sum, chunkFrames, limit) {
			break
		}
		trimMS += chunkMS
	}
	if trimMS > segLenMS {
		return segLenMS
	}
	return trimMS
}

func detectNonsilentPCM(pcm []int16, minSilenceMS int, silenceThresholdDB float64, seekStepMS int) [][2]int {
	silent := detectSilencePCM(pcm, minSilenceMS, silenceThresholdDB, seekStepMS)
	segLen := audioLenMS(len(pcm))
	if len(silent) == 0 {
		return [][2]int{{0, segLen}}
	}
	if silent[0][0] == 0 && silent[0][1] == segLen {
		return nil
	}
	prevEnd := 0
	nonsilent := make([][2]int, 0, len(silent)+1)
	var end int
	for _, r := range silent {
		nonsilent = append(nonsilent, [2]int{prevEnd, r[0]})
		prevEnd = r[1]
		end = r[1]
	}
	if end != segLen {
		nonsilent = append(nonsilent, [2]int{prevEnd, segLen})
	}
	if len(nonsilent) > 0 && nonsilent[0][0] == 0 && nonsilent[0][1] == 0 {
		nonsilent = nonsilent[1:]
	}
	return nonsilent
}

func detectSilencePCM(pcm []int16, minSilenceMS int, silenceThresholdDB float64, seekStepMS int) [][2]int {
	segLen := audioLenMS(len(pcm))
	if minSilenceMS <= 0 || seekStepMS <= 0 || segLen < minSilenceMS {
		return nil
	}
	limit := rmsLimitForInclusive(dbToAmplitude(silenceThresholdDB))
	windowFrames := msToFrames(minSilenceMS)
	prefix := squaredPrefixPCM(pcm)
	lastSliceStart := segLen - minSilenceMS
	silenceStarts := make([]int, 0, lastSliceStart/seekStepMS+2)
	for ms := 0; ms <= lastSliceStart; ms += seekStepMS {
		if rmsAtMost(windowSquaredSum(prefix, msToFrames(ms), windowFrames, len(pcm)), windowFrames, limit) {
			silenceStarts = append(silenceStarts, ms)
		}
	}
	if lastSliceStart%seekStepMS != 0 {
		ms := lastSliceStart
		if rmsAtMost(windowSquaredSum(prefix, msToFrames(ms), windowFrames, len(pcm)), windowFrames, limit) {
			silenceStarts = append(silenceStarts, ms)
		}
	}
	if len(silenceStarts) == 0 {
		return nil
	}
	ranges := make([][2]int, 0, len(silenceStarts))
	prev := silenceStarts[0]
	currentStart := prev
	for _, cur := range silenceStarts[1:] {
		continuous := cur == prev+seekStepMS
		silenceHasGap := cur > prev+minSilenceMS
		if !continuous && silenceHasGap {
			ranges = append(ranges, [2]int{currentStart, prev + minSilenceMS})
			currentStart = cur
		}
		prev = cur
	}
	ranges = append(ranges, [2]int{currentStart, prev + minSilenceMS})
	return ranges
}

func squaredPrefixPCM(pcm []int16) []int64 {
	prefix := make([]int64, len(pcm)+1)
	for i, v := range pcm {
		x := int64(v)
		prefix[i+1] = prefix[i] + x*x
	}
	return prefix
}

func windowSquaredSum(prefix []int64, startFrame, frameCount, totalFrames int) int64 {
	if startFrame < 0 {
		startFrame = 0
	}
	if startFrame > totalFrames {
		startFrame = totalFrames
	}
	end := startFrame + frameCount
	if end > totalFrames {
		end = totalFrames
	}
	return prefix[end] - prefix[startFrame]
}

func rmsAtMost(sum int64, frames int, limit int) bool {
	if frames <= 0 {
		return true
	}
	if limit < 0 {
		return sum == 0
	}
	next := int64(limit + 1)
	return sum < next*next*int64(frames)
}

func rmsLimitForInclusive(amplitude float64) int {
	return int(math.Floor(amplitude))
}

func rmsLimitForStrictLess(amplitude float64) int {
	return int(math.Ceil(amplitude)) - 1
}

func dbToAmplitude(db float64) float64 {
	return math.Pow(10, db/20) * 32768.0
}

func slicePCMByMS(pcm []int16, startMS, endMS int) []int16 {
	expected := slicePCMExpectedFrames(pcm, startMS, endMS)
	if expected == 0 {
		return nil
	}
	out := make([]int16, expected)
	copyPCMSlice(out, pcm, startMS, endMS)
	return out
}

func copyPCMSlice(dst, pcm []int16, startMS, endMS int) int {
	segLenMS := audioLenMS(len(pcm))
	if startMS < 0 {
		startMS = 0
	}
	if endMS < 0 {
		endMS = 0
	}
	if startMS > segLenMS {
		startMS = segLenMS
	}
	if endMS > segLenMS {
		endMS = segLenMS
	}
	if endMS < startMS {
		endMS = startMS
	}
	startFrame := msToFrames(startMS)
	endFrame := msToFrames(endMS)
	if startFrame > len(pcm) {
		startFrame = len(pcm)
	}
	if endFrame > len(pcm) {
		endFrame = len(pcm)
	}
	return copy(dst, pcm[startFrame:endFrame])
}

func slicePCMExpectedFrames(pcm []int16, startMS, endMS int) int {
	segLenMS := audioLenMS(len(pcm))
	if startMS < 0 {
		startMS = 0
	}
	if endMS < 0 {
		endMS = 0
	}
	if startMS > segLenMS {
		startMS = segLenMS
	}
	if endMS > segLenMS {
		endMS = segLenMS
	}
	if endMS < startMS {
		return 0
	}
	return msToFrames(endMS - startMS)
}

func reversePCM(pcm []int16) []int16 {
	out := make([]int16, len(pcm))
	for i := range pcm {
		out[len(pcm)-1-i] = pcm[i]
	}
	return out
}

func msToFrames(ms int) int {
	return ms * (Mono24kSampleRate / 1000)
}

func audioLenMS(frames int) int {
	if frames <= 0 {
		return 0
	}
	q := frames / 24
	r := frames % 24
	twice := r * 2
	switch {
	case twice < 24:
		return q
	case twice > 24:
		return q + 1
	case q%2 == 0:
		return q
	default:
		return q + 1
	}
}

func linspaceValue(start, stop float32, index, count int) float32 {
	if count <= 1 {
		return start
	}
	step := (stop - start) / float32(count-1)
	return start + float32(index)*step
}
