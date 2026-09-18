package jevlike

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	defaultTrainEpochs             = 8
	defaultTrainBatchSize          = 64
	defaultTrainLearningRate       = float32(2e-3)
	defaultTrainMaxGradNorm        = float32(1)
	defaultTrainSeed         int64 = 7
	adamBeta1                      = float32(0.9)
	adamBeta2                      = float32(0.999)
	adamEpsilon                    = float32(1e-8)
	adamWeightDecay                = float32(1e-4)
	embeddingInitStd               = float32(1) // nn.Embedding uses normal_(0, 1)
	positionInitStd                = float32(1)
)

// TrainConfig controls native TinyScorer optimisation.
type TrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float32
	Seed         int64
	MaxGradNorm  float32
}

// EpochMetrics records one training epoch.
type EpochMetrics struct {
	Epoch         int     `json:"epoch"`
	TrainNLL      float64 `json:"train_nll"`
	ValidationNLL float64 `json:"validation_nll"`
}

// TrainResult captures the best validation checkpoint and training history.
type TrainResult struct {
	History           []EpochMetrics       `json:"history"`
	BestValidationNLL float64              `json:"best_validation_nll"`
	BestState         map[string][]float32 `json:"best_state"`
}

type adamState struct {
	M []float32
	V []float32
}

// DefaultTrainConfig returns defaults matching the Python trainer.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Epochs:       defaultTrainEpochs,
		BatchSize:    defaultTrainBatchSize,
		LearningRate: defaultTrainLearningRate,
		Seed:         defaultTrainSeed,
		MaxGradNorm:  defaultTrainMaxGradNorm,
	}
}

// NewInitializedTinyScorer creates a scorer with deterministic non-zero weights.
func NewInitializedTinyScorer(config Config, seed int64) (*TinyScorer, error) {
	model, err := NewTinyScorer(config)
	if err != nil {
		return nil, err
	}
	if err := InitializeTinyScorer(model, seed); err != nil {
		return nil, err
	}
	return model, nil
}

// InitializeTinyScorer resets scorer parameters with deterministic non-zero values.
func InitializeTinyScorer(model *TinyScorer, seed int64) error {
	if err := model.Validate(); err != nil {
		return err
	}
	rng := rand.New(rand.NewSource(seed))
	for i := range model.EmbeddingWeight {
		model.EmbeddingWeight[i] = float32(rng.NormFloat64()) * embeddingInitStd
	}
	for i := 0; i < model.Config.Width; i++ {
		model.EmbeddingWeight[i] = 0
	}
	for i := range model.PositionWeight {
		model.PositionWeight[i] = float32(rng.NormFloat64()) * positionInitStd
	}
	for i := range model.Head.ContextNormWeight {
		model.Head.ContextNormWeight[i] = 1
		model.Head.ContextNormBias[i] = 0
		model.Head.OptionNormWeight[i] = 1
		model.Head.OptionNormBias[i] = 0
	}
	initLinearWeights(rng, model.Head.QueryWeight, model.Config.Width)
	initLinearWeights(rng, model.Head.KeyWeight, model.Config.Width)
	initLinearWeights(rng, model.Head.ValueWeight, model.Config.Width)
	return nil
}

// Backward differentiates TinyScorer.Forward, including embedding and position weights.
func (m *TinyScorer) Backward(batch ByteBatch, dLogits [][]float32) (map[string][]float32, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	rows, contextTokens, optionsPerRow, optionTokens, err := validateBatchForScorer(batch, m.Config)
	if err != nil {
		return nil, err
	}
	if len(dLogits) != rows {
		return nil, fmt.Errorf("jevlike scorer gradient row mismatch logits=%d want=%d", len(dLogits), rows)
	}
	for row := 0; row < rows; row++ {
		if len(dLogits[row]) != optionsPerRow {
			return nil, fmt.Errorf("jevlike scorer gradient shape mismatch row=%d logits=%d want=%d", row, len(dLogits[row]), optionsPerRow)
		}
	}

	context, options, optionCounts := encodeTinyBatch(m, batch, rows, contextTokens, optionsPerRow, optionTokens)
	headGrads, dContext, dOptions, err := m.Head.Backward(context, batch.ContextMask, options, batch.OptionMask, dLogits)
	if err != nil {
		return nil, err
	}

	dEmbedding := make([]float32, len(m.EmbeddingWeight))
	dPosition := make([]float32, len(m.PositionWeight))
	width := m.Config.Width
	for row := 0; row < rows; row++ {
		for token := 0; token < contextTokens; token++ {
			id := int(batch.ContextIDs[row][token])
			if id != 0 {
				embeddingBase := id * width
				for dim := 0; dim < width; dim++ {
					dEmbedding[embeddingBase+dim] += dContext[row][token][dim]
				}
			}
			positionBase := token * width
			for dim := 0; dim < width; dim++ {
				dPosition[positionBase+dim] += dContext[row][token][dim]
			}
		}
		for option := 0; option < optionsPerRow; option++ {
			count := optionCounts[row][option]
			if count == 0 {
				continue
			}
			scale := float32(1) / float32(count)
			for token := 0; token < optionTokens; token++ {
				if !batch.OptionTokenMask[row][option][token] {
					continue
				}
				id := int(batch.OptionIDs[row][option][token])
				if id == 0 {
					continue
				}
				embeddingBase := id * width
				for dim := 0; dim < width; dim++ {
					dEmbedding[embeddingBase+dim] += dOptions[row][option][dim] * scale
				}
			}
		}
	}

	grads := make(map[string][]float32, len(headGrads)+2)
	grads[embeddingParamName] = dEmbedding
	grads[positionParamName] = dPosition
	for name, values := range headGrads {
		grads[name] = values
	}
	return grads, nil
}

// TrainTinyScorer fits a native tiny scorer and restores the best validation state.
func TrainTinyScorer(model *TinyScorer, train, validation []ChoiceExample, cfg TrainConfig) (TrainResult, error) {
	cfg, err := cfg.resolved()
	if err != nil {
		return TrainResult{}, err
	}
	if model == nil {
		return TrainResult{}, fmt.Errorf("jevlike scorer is nil")
	}
	if len(train) == 0 {
		return TrainResult{}, fmt.Errorf("jevlike training set is empty")
	}
	if len(validation) == 0 {
		return TrainResult{}, fmt.Errorf("jevlike validation set is empty")
	}
	if needsTinyInitialization(model) {
		if err := InitializeTinyScorer(model, cfg.Seed); err != nil {
			return TrainResult{}, err
		}
	}

	params := model.NamedParameterMap()
	states := make(map[string]adamState, len(params))
	shuffle := rand.New(rand.NewSource(cfg.Seed))
	indices := make([]int, len(train))
	for i := range indices {
		indices[i] = i
	}

	result := TrainResult{
		History:           make([]EpochMetrics, 0, cfg.Epochs),
		BestValidationNLL: math.Inf(1),
	}
	step := 0
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		shuffle.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
		var trainTotal float64
		trainCount := 0
		for start := 0; start < len(indices); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(indices) {
				end = len(indices)
			}
			batchExamples := gatherExamples(train, indices[start:end])
			batch, err := BuildByteBatch(batchExamples, model.Config.ContextTokens, model.Config.OptionTokens)
			if err != nil {
				return TrainResult{}, err
			}
			logits, err := model.Forward(batch)
			if err != nil {
				return TrainResult{}, err
			}
			loss, dLogits, err := crossEntropyLossAndGradient(logits, batch.OptionMask, batch.Labels)
			if err != nil {
				return TrainResult{}, err
			}
			grads, err := model.Backward(batch, dLogits)
			if err != nil {
				return TrainResult{}, err
			}
			clipGradientMapByGlobalNorm(grads, cfg.MaxGradNorm)
			step++
			if err := adamwStep(params, grads, states, step, cfg.LearningRate); err != nil {
				return TrainResult{}, err
			}
			if err := model.LoadNamedParameters(params); err != nil {
				return TrainResult{}, err
			}
			trainTotal += loss * float64(len(batch.Labels))
			trainCount += len(batch.Labels)
		}
		validationLoss, err := meanCrossEntropyLoss(model, validation, cfg.BatchSize)
		if err != nil {
			return TrainResult{}, err
		}
		result.History = append(result.History, EpochMetrics{
			Epoch:         epoch,
			TrainNLL:      trainTotal / float64(trainCount),
			ValidationNLL: validationLoss,
		})
		if validationLoss < result.BestValidationNLL {
			result.BestValidationNLL = validationLoss
			result.BestState = cloneParameterMap(params)
		}
	}
	if result.BestState != nil {
		if err := model.LoadNamedParameters(result.BestState); err != nil {
			return TrainResult{}, err
		}
	}
	return result, nil
}

func (cfg TrainConfig) resolved() (TrainConfig, error) {
	out := cfg
	if out.Epochs == 0 {
		out.Epochs = defaultTrainEpochs
	}
	if out.BatchSize == 0 {
		out.BatchSize = defaultTrainBatchSize
	}
	if out.LearningRate == 0 {
		out.LearningRate = defaultTrainLearningRate
	}
	if out.MaxGradNorm == 0 {
		out.MaxGradNorm = defaultTrainMaxGradNorm
	}
	if out.Epochs <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike epochs must be positive")
	}
	if out.BatchSize <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike batch size must be positive")
	}
	if out.LearningRate <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike learning rate must be positive")
	}
	if out.MaxGradNorm <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike max grad norm must be positive")
	}
	return out, nil
}

func encodeTinyBatch(m *TinyScorer, batch ByteBatch, rows, contextTokens, optionsPerRow, optionTokens int) ([][][]float32, [][][]float32, [][]int) {
	context := make([][][]float32, rows)
	options := make([][][]float32, rows)
	optionCounts := make([][]int, rows)
	for row := 0; row < rows; row++ {
		context[row] = make([][]float32, contextTokens)
		for token := 0; token < contextTokens; token++ {
			vector := make([]float32, m.Config.Width)
			copy(vector, m.embeddingRow(int(batch.ContextIDs[row][token])))
			position := m.positionRow(token)
			for dim := 0; dim < m.Config.Width; dim++ {
				vector[dim] += position[dim]
			}
			context[row][token] = vector
		}
		options[row] = make([][]float32, optionsPerRow)
		optionCounts[row] = make([]int, optionsPerRow)
		for option := 0; option < optionsPerRow; option++ {
			vector := make([]float32, m.Config.Width)
			count := 0
			for token := 0; token < optionTokens; token++ {
				if !batch.OptionTokenMask[row][option][token] {
					continue
				}
				embedding := m.embeddingRow(int(batch.OptionIDs[row][option][token]))
				for dim := 0; dim < m.Config.Width; dim++ {
					vector[dim] += embedding[dim]
				}
				count++
			}
			if count > 1 {
				denominator := float32(count)
				for dim := 0; dim < m.Config.Width; dim++ {
					vector[dim] /= denominator
				}
			}
			options[row][option] = vector
			optionCounts[row][option] = count
		}
	}
	return context, options, optionCounts
}

func crossEntropyLossAndGradient(logits [][]float32, optionMask [][]bool, labels []int) (float64, [][]float32, error) {
	rows := len(logits)
	if rows == 0 {
		return 0, nil, fmt.Errorf("jevlike loss needs at least one row")
	}
	if len(optionMask) != rows || len(labels) != rows {
		return 0, nil, fmt.Errorf("jevlike loss shape mismatch logits=%d option_mask=%d labels=%d", len(logits), len(optionMask), len(labels))
	}
	dLogits := make([][]float32, rows)
	var total float64
	for row := 0; row < rows; row++ {
		if len(logits[row]) == 0 {
			return 0, nil, fmt.Errorf("jevlike loss row=%d has no options", row)
		}
		if len(optionMask[row]) != len(logits[row]) {
			return 0, nil, fmt.Errorf("jevlike loss row=%d mask=%d want=%d", row, len(optionMask[row]), len(logits[row]))
		}
		label := labels[row]
		if label < 0 || label >= len(logits[row]) || !optionMask[row][label] {
			return 0, nil, fmt.Errorf("jevlike loss row=%d label=%d out of range or masked", row, label)
		}
		found := false
		maxValue := float64(0)
		for option, active := range optionMask[row] {
			if !active {
				continue
			}
			value := float64(logits[row][option])
			if !found || value > maxValue {
				maxValue = value
				found = true
			}
		}
		if !found {
			return 0, nil, fmt.Errorf("jevlike loss row=%d has no active options", row)
		}
		var sumExp float64
		for option, active := range optionMask[row] {
			if !active {
				continue
			}
			sumExp += math.Exp(float64(logits[row][option]) - maxValue)
		}
		logSumExp := maxValue + math.Log(sumExp)
		total += logSumExp - float64(logits[row][label])
		dLogits[row] = make([]float32, len(logits[row]))
		for option, active := range optionMask[row] {
			if !active {
				continue
			}
			probability := float32(math.Exp(float64(logits[row][option]) - logSumExp))
			dLogits[row][option] = probability
		}
		dLogits[row][label] -= 1
	}
	invRows := float32(1) / float32(rows)
	for row := range dLogits {
		for option := range dLogits[row] {
			dLogits[row][option] *= invRows
		}
	}
	return total / float64(rows), dLogits, nil
}

func meanCrossEntropyLoss(model *TinyScorer, examples []ChoiceExample, batchSize int) (float64, error) {
	if len(examples) == 0 {
		return 0, fmt.Errorf("jevlike evaluation set is empty")
	}
	if batchSize <= 0 {
		return 0, fmt.Errorf("jevlike batch size must be positive")
	}
	var total float64
	count := 0
	for start := 0; start < len(examples); start += batchSize {
		end := start + batchSize
		if end > len(examples) {
			end = len(examples)
		}
		batch, err := BuildByteBatch(examples[start:end], model.Config.ContextTokens, model.Config.OptionTokens)
		if err != nil {
			return 0, err
		}
		logits, err := model.Forward(batch)
		if err != nil {
			return 0, err
		}
		loss, _, err := crossEntropyLossAndGradient(logits, batch.OptionMask, batch.Labels)
		if err != nil {
			return 0, err
		}
		total += loss * float64(len(batch.Labels))
		count += len(batch.Labels)
	}
	return total / float64(count), nil
}

func gatherExamples(examples []ChoiceExample, indices []int) []ChoiceExample {
	batch := make([]ChoiceExample, len(indices))
	for i, index := range indices {
		batch[i] = examples[index]
	}
	return batch
}

func clipGradientMapByGlobalNorm(grads map[string][]float32, maxNorm float32) float64 {
	if maxNorm <= 0 {
		return 0
	}
	var sumSquares float64
	for _, values := range grads {
		for _, value := range values {
			sumSquares += float64(value) * float64(value)
		}
	}
	norm := math.Sqrt(sumSquares)
	if norm == 0 || norm <= float64(maxNorm) {
		return norm
	}
	scale := maxNorm / float32(norm)
	for _, values := range grads {
		for i := range values {
			values[i] *= scale
		}
	}
	return norm
}

func adamwStep(params, grads map[string][]float32, states map[string]adamState, step int, learningRate float32) error {
	beta1Correction := float32(1 - math.Pow(float64(adamBeta1), float64(step)))
	beta2Correction := float32(1 - math.Pow(float64(adamBeta2), float64(step)))
	for name, values := range params {
		grad, ok := grads[name]
		if !ok {
			return fmt.Errorf("missing gradient for %q", name)
		}
		if len(grad) != len(values) {
			return fmt.Errorf("gradient length mismatch for %q: got %d want %d", name, len(grad), len(values))
		}
		state, ok := states[name]
		if !ok {
			state = adamState{M: make([]float32, len(values)), V: make([]float32, len(values))}
		}
		for i := range values {
			g := grad[i]
			state.M[i] = adamBeta1*state.M[i] + (1-adamBeta1)*g
			state.V[i] = adamBeta2*state.V[i] + (1-adamBeta2)*g*g
			mHat := state.M[i] / beta1Correction
			vHat := state.V[i] / beta2Correction
			values[i] -= learningRate * adamWeightDecay * values[i]
			values[i] -= learningRate * mHat / (float32(math.Sqrt(float64(vHat))) + adamEpsilon)
		}
		states[name] = state
	}
	return nil
}

func cloneParameterMap(params map[string][]float32) map[string][]float32 {
	cloned := make(map[string][]float32, len(params))
	for name, values := range params {
		cloned[name] = append([]float32(nil), values...)
	}
	return cloned
}

func needsTinyInitialization(model *TinyScorer) bool {
	if model == nil {
		return false
	}
	return allZeroFloat32(model.EmbeddingWeight) &&
		allZeroFloat32(model.PositionWeight) &&
		allZeroFloat32(model.Head.QueryWeight) &&
		allZeroFloat32(model.Head.KeyWeight) &&
		allZeroFloat32(model.Head.ValueWeight)
}

func allZeroFloat32(values []float32) bool {
	for _, value := range values {
		if value != 0 {
			return false
		}
	}
	return true
}

func initLinearWeights(rng *rand.Rand, weight []float32, fanIn int) {
	bound := 1 / math.Sqrt(float64(fanIn))
	for i := range weight {
		weight[i] = float32((rng.Float64()*2 - 1) * bound)
	}
}
