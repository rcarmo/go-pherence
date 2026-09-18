package jevlike

import (
	"fmt"
	"math"
	"math/rand"
)

func InitializeFrozenScorer(model *FrozenScorer, seed int64) error {
	if err := validateFrozenScorer(model, false); err != nil {
		return err
	}
	rng := rand.New(rand.NewSource(seed))
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

func TrainFrozenScorer(model *FrozenScorer, train, validation []ChoiceExample, cfg TrainConfig) (TrainResult, error) {
	cfg, err := cfg.resolved()
	if err != nil {
		return TrainResult{}, err
	}
	if err := validateFrozenScorer(model, true); err != nil {
		return TrainResult{}, err
	}
	if len(train) == 0 {
		return TrainResult{}, fmt.Errorf("jevlike training set is empty")
	}
	if len(validation) == 0 {
		return TrainResult{}, fmt.Errorf("jevlike validation set is empty")
	}
	if needsFrozenInitialization(model) {
		if err := InitializeFrozenScorer(model, cfg.Seed); err != nil {
			return TrainResult{}, err
		}
	}

	params := model.Head.NamedParameterMap(defaultHeadPrefix)
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
			batch, err := model.encodeBatch(batchExamples)
			if err != nil {
				return TrainResult{}, err
			}
			logits, err := model.forwardEncodedBatch(batch, false)
			if err != nil {
				return TrainResult{}, err
			}
			loss, dLogits, err := crossEntropyLossAndGradient(logits, batch.OptionMask, batch.Labels)
			if err != nil {
				return TrainResult{}, err
			}
			grads, _, _, err := model.Head.Backward(batch.Context, batch.ContextMask, batch.Options, batch.OptionMask, dLogits)
			if err != nil {
				return TrainResult{}, err
			}
			clipGradientMapByGlobalNorm(grads, cfg.MaxGradNorm)
			step++
			if err := adamwStep(params, grads, states, step, cfg.LearningRate); err != nil {
				return TrainResult{}, err
			}
			if err := model.Head.LoadNamedParameters(defaultHeadPrefix, params); err != nil {
				return TrainResult{}, err
			}
			trainTotal += loss * float64(len(batch.Labels))
			trainCount += len(batch.Labels)
		}
		validationLoss, err := meanCrossEntropyLossFrozen(model, validation, cfg.BatchSize)
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
		if err := model.Head.LoadNamedParameters(defaultHeadPrefix, result.BestState); err != nil {
			return TrainResult{}, err
		}
	}
	return result, nil
}

func EvaluateFrozen(m *FrozenScorer, examples []ChoiceExample, batchSize int) (Evaluation, error) {
	if len(examples) == 0 || batchSize <= 0 {
		return Evaluation{}, fmt.Errorf("evaluation requires examples and positive batch size")
	}
	var all, shuffled [][]float32
	labels := make([]int, 0, len(examples))
	for start := 0; start < len(examples); start += batchSize {
		end := min(start+batchSize, len(examples))
		batch, err := m.encodeBatch(examples[start:end])
		if err != nil {
			return Evaluation{}, err
		}
		normal, err := m.forwardEncodedBatch(batch, false)
		if err != nil {
			return Evaluation{}, err
		}
		control, err := m.forwardEncodedBatch(batch, true)
		if err != nil {
			return Evaluation{}, err
		}
		for i, ex := range examples[start:end] {
			all = append(all, normal[i][:len(ex.Options)])
			shuffled = append(shuffled, control[i][:len(ex.Options)])
			labels = append(labels, ex.Label)
		}
	}
	a, err := ComputeMetrics(all, labels)
	if err != nil {
		return Evaluation{}, err
	}
	b, err := ComputeMetrics(shuffled, labels)
	return Evaluation{Model: a, ShuffledContext: b}, err
}

func meanCrossEntropyLossFrozen(model *FrozenScorer, examples []ChoiceExample, batchSize int) (float64, error) {
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
		batch, err := model.encodeBatch(examples[start:end])
		if err != nil {
			return 0, err
		}
		logits, err := model.forwardEncodedBatch(batch, false)
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

func needsFrozenInitialization(model *FrozenScorer) bool {
	if model == nil {
		return false
	}
	return allZeroFloat32(model.Head.QueryWeight) &&
		allZeroFloat32(model.Head.KeyWeight) &&
		allZeroFloat32(model.Head.ValueWeight)
}
