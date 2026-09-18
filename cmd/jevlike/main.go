package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

const (
	defaultTinyWidth         = 64
	defaultTinyRank          = 64
	defaultTinyContextTokens = 192
	defaultTinyOptionTokens  = 32
)

var errHelpRequested = errors.New("help requested")

type optionList []string

func (o *optionList) String() string {
	return strings.Join(*o, ",")
}

func (o *optionList) Set(value string) error {
	*o = append(*o, value)
	return nil
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "jevlike:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		writeUsage(stderr)
		return fmt.Errorf("expected subcommand")
	}

	switch args[0] {
	case "frozen-train":
		return runFrozenTrain(args[1:], stdout, stderr)
	case "frozen-predict":
		return runFrozenPredict(args[1:], stdout, stderr)
	case "frozen-eval":
		return runFrozenEval(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		writeUsage(stdout)
		return nil
	case "synthetic":
		return runSynthetic(args[1:], stdout, stderr)
	case "wikispeedia":
		return runWikispeedia(args[1:], stdout, stderr)
	case "train":
		return runTrain(args[1:], stdout, stderr)
	case "predict":
		return runPredict(args[1:], stdout, stderr)
	case "eval":
		return runEval(args[1:], stdout, stderr)
	default:
		writeUsage(stderr)
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func writeUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage: jevlike <subcommand> [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  synthetic    write synthetic train/validation/test JSONL files")
	fmt.Fprintln(w, "  wikispeedia  build train/validation/test JSONL files from Wikispeedia")
	fmt.Fprintln(w, "  train        train a native tiny scorer and save the best checkpoint")
	fmt.Fprintln(w, "  predict      score one context with repeated -option flags (tiny checkpoints only)")
	fmt.Fprintln(w, "  eval         evaluate a tiny checkpoint on JSONL data and print JSON metrics")
	fmt.Fprintln(w, "  frozen-train / frozen-predict / frozen-eval: frozen local decoder workflows")
}

func runSynthetic(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("synthetic", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var outputDir string
	var trainCount, validationCount, testCount int
	var seed int64
	fs.StringVar(&outputDir, "output-dir", "", "Directory for train.jsonl, validation.jsonl and test.jsonl")
	fs.IntVar(&trainCount, "train", 1024, "Number of synthetic training examples")
	fs.IntVar(&validationCount, "validation", 128, "Number of synthetic validation examples")
	fs.IntVar(&testCount, "test", 128, "Number of synthetic test examples")
	fs.Int64Var(&seed, "seed", 7, "Synthetic dataset seed")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if errors.Is(err, errHelpRequested) {
			return nil
		}
		return err
	}
	if outputDir == "" {
		return fmt.Errorf("-output-dir is required")
	}

	sizes := jevlike.SplitSizes{Train: trainCount, Validation: validationCount, Test: testCount}
	if err := jevlike.WriteSyntheticDataset(outputDir, sizes, seed); err != nil {
		return err
	}
	return writeJSON(stdout, struct {
		OutputDir string             `json:"output_dir"`
		Seed      int64              `json:"seed"`
		Sizes     jevlike.SplitSizes `json:"sizes"`
	}{OutputDir: outputDir, Seed: seed, Sizes: sizes})
}

func runWikispeedia(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("wikispeedia", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var rootDir, outputDir string
	var maxOptions int
	fs.StringVar(&rootDir, "root-dir", "", "Wikispeedia root containing wikispeedia_paths-and-graph and plaintext_articles")
	fs.StringVar(&outputDir, "output-dir", "", "Directory for train.jsonl, validation.jsonl and test.jsonl")
	fs.IntVar(&maxOptions, "max-options", 64, "Maximum answer options per example")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if errors.Is(err, errHelpRequested) {
			return nil
		}
		return err
	}
	if rootDir == "" || outputDir == "" {
		return fmt.Errorf("-root-dir and -output-dir are required")
	}

	sizes, err := jevlike.BuildWikispeedia(rootDir, outputDir, jevlike.WikispeediaOptions{MaxOptions: maxOptions})
	if err != nil {
		return err
	}
	return writeJSON(stdout, struct {
		RootDir   string             `json:"root_dir"`
		OutputDir string             `json:"output_dir"`
		Sizes     jevlike.SplitSizes `json:"sizes"`
	}{RootDir: rootDir, OutputDir: outputDir, Sizes: sizes})
}

func runTrain(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaults := jevlike.DefaultTrainConfig()

	var trainPath, validationPath, checkpointPath string
	var width, rank, contextTokens, optionTokens int
	var epochs, batchSize int
	var learningRate, maxGradNorm float64
	var seed int64

	fs.StringVar(&trainPath, "train", "", "Training JSONL file")
	fs.StringVar(&validationPath, "validation", "", "Validation JSONL file")
	fs.StringVar(&checkpointPath, "checkpoint", "", "Output checkpoint path")
	fs.IntVar(&width, "width", defaultTinyWidth, "Tiny scorer width")
	fs.IntVar(&rank, "rank", defaultTinyRank, "Attention rank")
	fs.IntVar(&contextTokens, "context-tokens", defaultTinyContextTokens, "Maximum context byte tokens")
	fs.IntVar(&optionTokens, "option-tokens", defaultTinyOptionTokens, "Maximum option byte tokens")
	fs.IntVar(&epochs, "epochs", defaults.Epochs, "Training epochs")
	fs.IntVar(&batchSize, "batch-size", defaults.BatchSize, "Training batch size")
	fs.Float64Var(&learningRate, "learning-rate", float64(defaults.LearningRate), "AdamW learning rate")
	fs.Int64Var(&seed, "seed", defaults.Seed, "Initialization and shuffle seed")
	fs.Float64Var(&maxGradNorm, "max-grad-norm", float64(defaults.MaxGradNorm), "Global gradient clipping norm")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if errors.Is(err, errHelpRequested) {
			return nil
		}
		return err
	}
	if trainPath == "" || validationPath == "" || checkpointPath == "" {
		return fmt.Errorf("-train, -validation and -checkpoint are required")
	}

	trainExamples, err := jevlike.LoadJSONL(trainPath)
	if err != nil {
		return err
	}
	validationExamples, err := jevlike.LoadJSONL(validationPath)
	if err != nil {
		return err
	}

	config := jevlike.Config{
		Width:         width,
		Rank:          rank,
		ContextTokens: contextTokens,
		OptionTokens:  optionTokens,
	}
	model, err := jevlike.NewInitializedTinyScorer(config, seed)
	if err != nil {
		return err
	}
	result, err := jevlike.TrainTinyScorer(model, trainExamples, validationExamples, jevlike.TrainConfig{
		Epochs:       epochs,
		BatchSize:    batchSize,
		LearningRate: float32(learningRate),
		Seed:         seed,
		MaxGradNorm:  float32(maxGradNorm),
	})
	if err != nil {
		return err
	}
	checkpoint, err := jevlike.TinyCheckpoint(model)
	if err != nil {
		return err
	}
	if err := jevlike.SaveCheckpoint(checkpointPath, checkpoint); err != nil {
		return err
	}
	return writeJSON(stdout, struct {
		Train             string                 `json:"train"`
		Validation        string                 `json:"validation"`
		Checkpoint        string                 `json:"checkpoint"`
		Config            jevlike.Config         `json:"config"`
		History           []jevlike.EpochMetrics `json:"history"`
		BestValidationNLL float64                `json:"best_validation_nll"`
	}{
		Train:             trainPath,
		Validation:        validationPath,
		Checkpoint:        checkpointPath,
		Config:            config,
		History:           result.History,
		BestValidationNLL: result.BestValidationNLL,
	})
}

func runPredict(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var checkpointPath, context string
	var options optionList
	fs.StringVar(&checkpointPath, "checkpoint", "", "Input tiny checkpoint path")
	fs.StringVar(&context, "context", "", "Example context")
	fs.Var(&options, "option", "Answer option (repeat for each choice)")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if errors.Is(err, errHelpRequested) {
			return nil
		}
		return err
	}
	if checkpointPath == "" || context == "" {
		return fmt.Errorf("-checkpoint and -context are required")
	}
	if len(options) < 2 {
		return fmt.Errorf("at least two -option flags are required")
	}

	model, err := loadTinyModel(checkpointPath)
	if err != nil {
		return err
	}
	batch, err := jevlike.BuildByteBatch([]jevlike.ChoiceExample{{
		Context: context,
		Options: append([]string(nil), options...),
		Label:   0,
	}}, model.Config.ContextTokens, model.Config.OptionTokens)
	if err != nil {
		return err
	}
	prediction, err := model.Predict(batch)
	if err != nil {
		return err
	}
	logits := append([]float32(nil), prediction.Logits[0][:len(options)]...)
	probabilities := append([]float32(nil), prediction.Probabilities[0][:len(options)]...)
	bestIndex := 0
	for i := 1; i < len(probabilities); i++ {
		if probabilities[i] > probabilities[bestIndex] {
			bestIndex = i
		}
	}
	return writeJSON(stdout, struct {
		Options       []string  `json:"options"`
		Logits        []float32 `json:"logits"`
		Probabilities []float32 `json:"probabilities"`
		BestIndex     int       `json:"best_index"`
		BestOption    string    `json:"best_option"`
	}{
		Options:       append([]string(nil), options...),
		Logits:        logits,
		Probabilities: probabilities,
		BestIndex:     bestIndex,
		BestOption:    options[bestIndex],
	})
}

func runEval(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaults := jevlike.DefaultTrainConfig()
	var checkpointPath, dataPath string
	var batchSize int
	fs.StringVar(&checkpointPath, "checkpoint", "", "Input tiny checkpoint path")
	fs.StringVar(&dataPath, "data", "", "Evaluation JSONL file")
	fs.IntVar(&batchSize, "batch-size", defaults.BatchSize, "Evaluation batch size")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if errors.Is(err, errHelpRequested) {
			return nil
		}
		return err
	}
	if checkpointPath == "" || dataPath == "" {
		return fmt.Errorf("-checkpoint and -data are required")
	}

	model, err := loadTinyModel(checkpointPath)
	if err != nil {
		return err
	}
	examples, err := jevlike.LoadJSONL(dataPath)
	if err != nil {
		return err
	}
	evaluation, err := jevlike.EvaluateTiny(model, examples, batchSize)
	if err != nil {
		return err
	}
	return writeJSON(stdout, evaluation)
}

func parseSubcommandFlags(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return errHelpRequested
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", fs.Args())
	}
	return nil
}

func loadTinyModel(path string) (*jevlike.TinyScorer, error) {
	checkpoint, err := jevlike.LoadCheckpoint(path)
	if err != nil {
		return nil, err
	}
	if checkpoint.Encoder != "tiny" {
		return nil, fmt.Errorf("checkpoint encoder %q is unsupported; only tiny checkpoints work with this CLI", checkpoint.Encoder)
	}
	return checkpoint.Tiny()
}

func writeJSON(w io.Writer, value any) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}
