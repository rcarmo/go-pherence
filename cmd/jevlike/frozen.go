package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"path/filepath"

	loadertokenizer "github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

const (
	defaultFrozenRank          = 64
	defaultFrozenContextTokens = 192
	defaultFrozenOptionTokens  = 32
)

type frozenEncoderLoaderFunc func(dir string, bos int) (jevlike.TokenEncoder, int, string, error)

var frozenEncoderLoader frozenEncoderLoaderFunc = loadLocalFrozenEncoder

func runFrozenTrain(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("frozen-train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaults := jevlike.DefaultTrainConfig()

	var encoderModelDir, trainPath, validationPath, checkpointPath string
	var encoderBOS int
	var rank, contextTokens, optionTokens int
	var epochs, batchSize int
	var learningRate, maxGradNorm float64
	var seed int64

	fs.StringVar(&encoderModelDir, "encoder-model", "", "Local safetensors model directory with tokenizer.json")
	fs.StringVar(&trainPath, "train", "", "Training JSONL file")
	fs.StringVar(&validationPath, "validation", "", "Validation JSONL file")
	fs.StringVar(&checkpointPath, "checkpoint", "", "Output frozen checkpoint path")
	fs.IntVar(&encoderBOS, "encoder-bos", -1, "Optional explicit BOS token ID to prepend before tokenizer output")
	fs.IntVar(&rank, "rank", defaultFrozenRank, "Frozen scorer attention rank")
	fs.IntVar(&contextTokens, "context-tokens", defaultFrozenContextTokens, "Maximum context tokens")
	fs.IntVar(&optionTokens, "option-tokens", defaultFrozenOptionTokens, "Maximum option tokens")
	fs.IntVar(&epochs, "epochs", defaults.Epochs, "Training epochs")
	fs.IntVar(&batchSize, "batch-size", defaults.BatchSize, "Training batch size")
	fs.Float64Var(&learningRate, "learning-rate", float64(defaults.LearningRate), "AdamW learning rate")
	fs.Int64Var(&seed, "seed", defaults.Seed, "Initialization and shuffle seed")
	fs.Float64Var(&maxGradNorm, "max-grad-norm", float64(defaults.MaxGradNorm), "Global gradient clipping norm")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if encoderModelDir == "" || trainPath == "" || validationPath == "" || checkpointPath == "" {
		return fmt.Errorf("--encoder-model, --train, --validation and --checkpoint are required")
	}

	encoder, width, reference, err := frozenEncoderLoader(encoderModelDir, encoderBOS)
	if err != nil {
		return err
	}
	if width <= 0 {
		return fmt.Errorf("encoder width must be positive, got %d", width)
	}
	trainExamples, err := jevlike.LoadJSONL(trainPath)
	if err != nil {
		return err
	}
	validationExamples, err := jevlike.LoadJSONL(validationPath)
	if err != nil {
		return err
	}

	config := jevlike.Config{Width: width, Rank: rank, ContextTokens: contextTokens, OptionTokens: optionTokens}
	head, err := jevlike.NewAttentionHead(config.Width, config.Rank)
	if err != nil {
		return err
	}
	scorer := &jevlike.FrozenScorer{Config: config, Reference: reference, Head: *head, Encoder: encoder}
	if err := jevlike.InitializeFrozenScorer(scorer, seed); err != nil {
		return err
	}
	result, err := jevlike.TrainFrozenScorer(scorer, trainExamples, validationExamples, jevlike.TrainConfig{
		Epochs:       epochs,
		BatchSize:    batchSize,
		LearningRate: float32(learningRate),
		Seed:         seed,
		MaxGradNorm:  float32(maxGradNorm),
	})
	if err != nil {
		return err
	}
	checkpoint, err := jevlike.FrozenCheckpoint(scorer)
	if err != nil {
		return err
	}
	if err := jevlike.SaveCheckpoint(checkpointPath, checkpoint); err != nil {
		return err
	}
	return writeJSON(stdout, struct {
		EncoderModel      string                 `json:"encoder_model"`
		Train             string                 `json:"train"`
		Validation        string                 `json:"validation"`
		Checkpoint        string                 `json:"checkpoint"`
		Config            jevlike.Config         `json:"config"`
		History           []jevlike.EpochMetrics `json:"history"`
		BestValidationNLL float64                `json:"best_validation_nll"`
	}{
		EncoderModel:      encoderModelDir,
		Train:             trainPath,
		Validation:        validationPath,
		Checkpoint:        checkpointPath,
		Config:            config,
		History:           result.History,
		BestValidationNLL: result.BestValidationNLL,
	})
}

func runFrozenPredict(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("frozen-predict", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var encoderModelDir, checkpointPath, context string
	var encoderBOS int
	var options optionList
	fs.StringVar(&encoderModelDir, "encoder-model", "", "Local safetensors model directory with tokenizer.json")
	fs.StringVar(&checkpointPath, "checkpoint", "", "Input frozen checkpoint path")
	fs.StringVar(&context, "context", "", "Example context")
	fs.IntVar(&encoderBOS, "encoder-bos", -1, "Optional explicit BOS token ID to prepend before tokenizer output")
	fs.Var(&options, "option", "Answer option (repeat for each choice)")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if encoderModelDir == "" || checkpointPath == "" || context == "" {
		return fmt.Errorf("--encoder-model, --checkpoint and --context are required")
	}
	if len(options) < 2 {
		return fmt.Errorf("at least two --option flags are required")
	}

	scorer, err := loadFrozenScorer(checkpointPath, encoderModelDir, encoderBOS)
	if err != nil {
		return err
	}
	logitsBatch, err := scorer.Forward([]jevlike.ChoiceExample{{
		Context: context,
		Options: append([]string(nil), options...),
		Label:   0,
	}}, false)
	if err != nil {
		return err
	}
	logits := append([]float32(nil), logitsBatch[0][:len(options)]...)
	probabilities := stableSoftmaxFloat32(logits)
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

func runFrozenEval(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("frozen-eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaults := jevlike.DefaultTrainConfig()
	var encoderModelDir, checkpointPath, dataPath string
	var encoderBOS, batchSize int
	fs.StringVar(&encoderModelDir, "encoder-model", "", "Local safetensors model directory with tokenizer.json")
	fs.StringVar(&checkpointPath, "checkpoint", "", "Input frozen checkpoint path")
	fs.StringVar(&dataPath, "data", "", "Evaluation JSONL file")
	fs.IntVar(&encoderBOS, "encoder-bos", -1, "Optional explicit BOS token ID to prepend before tokenizer output")
	fs.IntVar(&batchSize, "batch-size", defaults.BatchSize, "Evaluation batch size")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if encoderModelDir == "" || checkpointPath == "" || dataPath == "" {
		return fmt.Errorf("--encoder-model, --checkpoint and --data are required")
	}

	scorer, err := loadFrozenScorer(checkpointPath, encoderModelDir, encoderBOS)
	if err != nil {
		return err
	}
	examples, err := jevlike.LoadJSONL(dataPath)
	if err != nil {
		return err
	}
	evaluation, err := jevlike.EvaluateFrozen(scorer, examples, batchSize)
	if err != nil {
		return err
	}
	return writeJSON(stdout, evaluation)
}

func loadFrozenScorer(checkpointPath, encoderModelDir string, encoderBOS int) (*jevlike.FrozenScorer, error) {
	encoder, width, _, err := frozenEncoderLoader(encoderModelDir, encoderBOS)
	if err != nil {
		return nil, err
	}
	checkpoint, err := jevlike.LoadCheckpoint(checkpointPath)
	if err != nil {
		return nil, err
	}
	if checkpoint.Encoder != "frozen" {
		return nil, fmt.Errorf("checkpoint encoder %q is not frozen", checkpoint.Encoder)
	}
	if width != checkpoint.Config.Width {
		return nil, fmt.Errorf("encoder width %d does not match checkpoint width %d", width, checkpoint.Config.Width)
	}
	return checkpoint.Frozen(encoder)
}

func loadLocalFrozenEncoder(dir string, bos int) (jevlike.TokenEncoder, int, string, error) {
	model, err := backbone.LoadLlama(dir)
	if err != nil {
		return nil, 0, "", err
	}
	tok, err := loadertokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, 0, "", err
	}
	width := model.Config.HiddenSize
	if width <= 0 {
		return nil, 0, "", fmt.Errorf("loaded encoder has invalid hidden size %d", width)
	}
	encoder := jevlike.DecoderEncoder{
		Model: model,
		Tokenize: func(text string) ([]int, error) {
			ids := tok.Encode(text)
			if bos < 0 {
				return ids, nil
			}
			out := make([]int, 0, len(ids)+1)
			out = append(out, bos)
			out = append(out, ids...)
			return out, nil
		},
	}
	return encoder, width, filepath.Clean(dir), nil
}

func stableSoftmaxFloat32(values []float32) []float32 {
	if len(values) == 0 {
		return nil
	}
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	out := make([]float32, len(values))
	var sum float64
	for i, value := range values {
		expValue := math.Exp(float64(value - maxValue))
		out[i] = float32(expValue)
		sum += expValue
	}
	if sum == 0 {
		return out
	}
	inv := float32(1 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}
