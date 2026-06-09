package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func configFor(size string) whisper.Config {
	switch size {
	case "tiny":
		return whisper.Tiny()
	case "base":
		return whisper.Base()
	case "small":
		return whisper.Small()
	case "medium":
		return whisper.Medium()
	case "large-v3":
		return whisper.LargeV3()
	case "turbo", "large-v3-turbo":
		return whisper.LargeV3Turbo()
	default:
		fmt.Fprintf(os.Stderr, "unknown model size %q\n", size)
		os.Exit(2)
	}
	return whisper.LargeV3Turbo()
}

func main() {
	modelPath := flag.String("model", "", "path to Whisper model.safetensors")
	audioPath := flag.String("audio", "", "path to WAV audio")
	size := flag.String("size", "turbo", "model size: tiny, base, small, medium, large-v3, turbo")
	layers := flag.Int("layers", 0, "max encoder layers to diagnose; 0 means all")
	flag.Parse()
	if *modelPath == "" || *audioPath == "" {
		flag.Usage()
		os.Exit(2)
	}

	cfg := configFor(*size)
	enc, _, err := whisper.LoadModel(*modelPath, cfg)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	samples, sr, err := audio.WAV(*audioPath)
	if err != nil {
		log.Fatalf("load audio: %v", err)
	}
	if sr != 16000 {
		samples = audio.ResampleSinc(samples, sr, 16000)
	}
	melCfg := audio.MelConfig{SampleRate: 16000, FFTSize: 400, HopLength: 160, NumMels: cfg.NumMelBins, NFFTPadded: 512}
	mel := audio.MelSpectrogram(samples, melCfg)
	if len(mel) == 0 {
		log.Fatal("empty mel spectrogram")
	}
	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:(m+1)*T], mel[m])
	}

	diags := enc.DiagnoseFFN(melFlat, T, *layers)
	fmt.Printf("layer\tvariant\tok\tseq\tcompared\tmax_abs\tmean_abs\trmse\trel_rmse\tcosine\n")
	for _, d := range diags {
		fmt.Printf("%d\t%s\t%t\t%d\t%d\t%.7g\t%.7g\t%.7g\t%.7g\t%.9f\n", d.Layer, d.Variant, d.OK, d.SeqLen, d.Compared, d.MaxAbs, d.MeanAbs, d.RMSE, d.RelRMSE, d.Cosine)
	}
}
