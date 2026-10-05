package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingStepSamples freezes all stochastic LSD inputs for deterministic
// parity. Arrays are frame-major; only valid Mask rows are consumed.
type TrainingStepSamples struct {
	Mask               []bool
	Noise              []float32
	DiagonalTime       []float32
	DistillS, DistillT []float32
}

// TrainingStepConfig controls the exact upstream reduction weights. Use
// DefaultTrainingStepConfig for p_equal=0.75 and eos_loss_weight=0.1.
type TrainingStepConfig struct {
	PEqual, EOSLossWeight float32
}

func DefaultTrainingStepConfig() TrainingStepConfig {
	return TrainingStepConfig{PEqual: 0.75, EOSLossWeight: 0.1}
}

type TrainingStepMetrics struct {
	FlowDiagonal, FlowDistill float64
	FlowLoss, EOS, Loss       float64
}

// TrainingStepGradients joins the backbone/conditioning, shared flow-head and
// per-frame learned log-variance leaf gradients.
type TrainingStepGradients struct {
	FlowLM              *FlowLMTrainingGradients
	Flow                *FlowHeadGradients
	Inputs              FlowLMTrainingInputGradients
	DiagonalLogVariance []float32
	DistillLogVariance  []float32
	Weighting           *LSDWeightGradients
}

// PocketTrainingStep evaluates one deterministic one-row upstream training
// step: EOS + normalized LSD diagonal + normalized minimal-stop-gradient s->t.
// Mimi latents, noise, times and log-variance outputs are supplied explicitly.
func PocketTrainingStep(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, batch FlowLMTrainingBatch, samples TrainingStepSamples, config TrainingStepConfig) (TrainingStepMetrics, TrainingStepGradients, error) {
	workspace, err := NewTrainingWorkspace(flowLM, flow, batch)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	return PocketTrainingStepInto(flowLM, flow, weighting, batch, samples, config, workspace)
}

func PocketTrainingStepInto(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, batch FlowLMTrainingBatch, samples TrainingStepSamples, config TrainingStepConfig, workspace *TrainingWorkspace) (TrainingStepMetrics, TrainingStepGradients, error) {
	return pocketTrainingStepIntoTapes(flowLM, flow, weighting, batch, samples, config, workspace, true)
}

// pocketTrainingStepIntoTapes keeps the owned-tape reference available to
// differential tests. Production always uses the request-owned row scratch.
func pocketTrainingStepIntoTapes(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, batch FlowLMTrainingBatch, samples TrainingStepSamples, config TrainingStepConfig, workspace *TrainingWorkspace, reuseTapes bool) (stepMetrics TrainingStepMetrics, stepGradients TrainingStepGradients, stepErr error) {
	// A live producer spans row consumers only; fence any later error before
	// cleanup so uncertain work cannot release producer storage.
	if b, ok := flowBackendForHandoff(flow); ok {
		defer func() {
			if b.handoffLease == nil {
				return
			}
			if stepErr != nil {
				b.fail(stepErr)
				return
			}
			if err := b.closeFlowHandoff(); err != nil {
				stepErr = err
				stepMetrics = TrainingStepMetrics{}
				stepGradients = TrainingStepGradients{}
			}
		}()
	}
	if flowLM == nil || flow == nil || len(flow.Time) != 2 || weighting == nil || workspace == nil || batch.Frames <= 0 || len(samples.Mask) != batch.Frames || !isFinite(config.PEqual) || config.PEqual < 0 || config.PEqual > 1 || !isFinite(config.EOSLossWeight) || config.EOSLossWeight < 0 {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS training step")
	}
	if flow.StepReductionBackend != nil && (flow.TrainingBackend == nil || flow.BranchBackend == nil || flow.LossBackend == nil) {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step reduction requires flow/branch/loss backends")
	}
	if err := validateTrainingStepSamples(batch, samples, flowLM.Hidden, flowLM.LatentDim); err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	_, ok := checked.MulInt(4, batch.Frames)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS weighting shape")
	}
	weightRows, ok := checked.MulInt(2, batch.Frames)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS weighting rows")
	}
	_, ok = checked.MulInt(batch.Frames, flowLM.Hidden)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS training hidden shape")
	}
	if workspace.frames != batch.Frames || workspace.hidden != flowLM.Hidden || workspace.latent != flowLM.LatentDim || workspace.weightRows != weightRows || !workspace.compatible(flowLM, flow) {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("Pocket TTS training workspace shape mismatch")
	}
	if workspace.zeroGradientOwner != nil && weighting.TrainingBackend != workspace.zeroGradientOwner {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("shared zero-gradient weighting owner changed")
	}
	if owner := workspace.zeroGradientOwner; owner != nil {
		if owner.forwardZeroWorkspace != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("forward zero workspace already borrowed")
		}
		owner.forwardZeroWorkspace = workspace
		defer func() { owner.forwardZeroWorkspace = nil }()
	}
	workspace.reset()
	if flow.Condition.In > len(workspace.dZ) || len(flow.Time) > len(workspace.weightInputs) || flow.Input.In > len(workspace.xTime) || flow.Final.Linear.Out > len(workspace.desired) {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("Pocket TTS training workspace validation shape mismatch")
	}
	if err := validateTrainableFlowHead(flow, workspace.dZ[:flow.Condition.In], workspace.weightInputs[:len(flow.Time)], workspace.xTime[:flow.Input.In], workspace.desired[:flow.Final.Linear.Out]); err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	weightInputs := workspace.weightInputs
	weightBackend := weighting.TrainingBackend
	sampleBackend, packed := weightBackend.(TrainingSampleWeightBackend)
	packed = packed && sampleBackend.SampleWeightEnabled()
	weightRoots, rootsOnly := weightBackend.(trainingWeightRootsBackend)
	rootsOnly = rootsOnly && weightRoots.weightRootsEnabled()
	stepWeight, combinedWeight := weightBackend.(trainingStepWeightBackend)
	combinedWeight = combinedWeight && stepWeight.stepWeightEnabled()
	stepBackbone, combinedBackbone := weightBackend.(trainingStepBackboneBackend)
	combinedBackbone = combinedBackbone && stepBackbone.stepBackboneEnabled()
	stepResultPolicy, reducedResults := weightBackend.(interface{ stepResultsEnabled() bool })
	reducedResults = reducedResults && stepResultPolicy.stepResultsEnabled()
	if reducedResults && !combinedBackbone {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step results requires composed backbone")
	}
	var combinedInputs FlowLMTrainingInputGradients
	if rootsOnly && !packed {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("weight roots require sample packing")
	}
	if !packed {
		for row := 0; row < batch.Frames; row++ {
			weightInputs[2*row], weightInputs[2*row+1] = samples.DiagonalTime[row], samples.DiagonalTime[row]
			base := 2*batch.Frames + 2*row
			weightInputs[base], weightInputs[base+1] = samples.DistillS[row], samples.DistillT[row]
		}
	}
	var weightTape *lsdWeightTape
	var logvars []float32
	var err error
	if weightBackend != nil {
		var result WeightTrainingResult
		var e error
		if packed {
			result, e = sampleBackend.EvaluateSampleWeight(weighting, WeightSampleInputs{samples.DiagonalTime, samples.DistillS, samples.DistillT, workspace.dDiagLogvar, workspace.dDistillLogvar}, false)
		} else {
			result, e = weightBackend.EvaluateWeight(weighting, weightInputs, make([]float32, weightRows), weightRows, false)
		}
		if e != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, e
		}
		if e := validateWeightBoundary(result, weightRows, weightInputs, samples.DiagonalTime, samples.DistillS, samples.DistillT); e != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, e
		}
		logvars = result.Output
	} else {
		weightTape, logvars, err = weighting.forward(weightInputs, weightRows)
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
	}
	zeroZ, zeroEOS := workspace.zeroZ, workspace.zeroEOS
	fusion, fusedEOS := flowLM.EOSBackend.(TrainingEOSProjectionBackend)
	fusedEOS = fusedEOS && fusion.EOSProjectionEnabled()
	if fusedEOS && flowLM.ProjectionBackend == nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("fused EOS requires projection owner")
	}
	var flowLMTape *flowLMTrainingTape
	var eosLoss float64
	privateForward, combinedForward := flowLM.EOSBackend.(*ResidentTrainingBackend)
	combinedForward = combinedForward && privateForward.eosForwardEnabled()
	if combinedForward {
		if !fusedEOS {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("EOS forward requires fused EOS")
		}
		if privateForward.forwardResultsEnabled() {
			flowLMTape, eosLoss, err = privateForward.evaluateForwardResults(flowLM, batch, samples.Mask, config.EOSLossWeight)
		} else {
			flowLMTape, eosLoss, err = privateForward.evaluateEOSForward(flowLM, batch, samples.Mask, config.EOSLossWeight)
		}
	} else {
		flowLMTape, err = flowLM.forwardTrainingTapeEOSMode(batch, zeroZ, zeroEOS, false, fusedEOS)
	}
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	forward := flowLMTape.output
	var dEOS []float32
	eosBackbone, combineEOS := flowLM.EOSBackend.(TrainingEOSBackboneBackend)
	combineEOS = combineEOS && eosBackbone.EOSBackboneEnabled()
	if combinedForward {
		if !combineEOS || flowLMTape.eosBackbone == nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("EOS forward missing backward policy")
		}
		dEOS = zeroEOS
	} else if combineEOS {
		if !fusedEOS || flowLMTape.backbone == nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("EOS backbone requires fused backbone")
		}
		eosLoss, err = eosBackbone.EvaluateEOSBackboneLoss(flowLM, forward.Z, samples.Mask, config.EOSLossWeight)
		if err == nil {
			flowLMTape.eosBackbone = eosBackbone
			flowLMTape.eosMask = append([]bool(nil), samples.Mask...)
			flowLMTape.eosWeight = config.EOSLossWeight
			dEOS = zeroEOS
		}
	} else if fusedEOS {
		var r EOSProjectionTrainingResult
		r, err = fusion.EvaluateEOSProjection(flowLM, forward.Z, samples.Mask, config.EOSLossWeight)
		if err == nil {
			err = validateEOSProjectionBoundary(r, batch.Frames, flowLM.Hidden, forward.Z)
		}
		if err == nil {
			eosLoss = r.Loss
			flowLMTape.fusedEOSInput = r.DInput
			dEOS = zeroEOS
		}
	} else if flowLM.EOSBackend != nil {
		eosLoss, dEOS, err = flowLM.EOSBackend.EvaluateEOS(flowLM, forward.EOS, samples.Mask, config.EOSLossWeight)
		if err == nil {
			if !finite64(eosLoss) || eosLoss < 0 {
				err = fmt.Errorf("invalid EOS backend scalar loss")
			} else {
				_, _, err = validateEOSBoundary([][]float32{{float32(eosLoss)}, dEOS}, forward.EOS, samples.Mask)
			}
		}
	} else {
		eosLoss, dEOS, err = EOSLossAndGradient(forward.EOS, samples.Mask)
		for i := range dEOS {
			dEOS[i] *= config.EOSLossWeight
		}
	}
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	valid := 0
	for _, active := range samples.Mask {
		if active {
			valid++
		}
	}
	if valid == 0 {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("Pocket TTS training step has no valid flow rows")
	}
	invValid := float32(1) / float32(valid)
	dZ, directTarget := workspace.dZ, workspace.directTarget
	flowGradients, discardFlowGradients := workspace.flowGradients, workspace.discardFlowGradients
	dDiagLogvar, dDistillLogvar := workspace.dDiagLogvar, workspace.dDistillLogvar
	metrics := TrainingStepMetrics{EOS: eosLoss}
	diagonalObjective, distillObjective := float64(0), float64(0)
	c, h := flowLM.LatentDim, flowLM.Hidden
	stepBackend := flow.StepReductionBackend
	var reduction StepReductionInputs
	if stepBackend != nil {
		reduction = StepReductionInputs{Frames: batch.Frames, Hidden: h, Latent: c, Mask: samples.Mask, EOS: eosLoss, DiagonalCondition: make([]float32, batch.Frames*h), DistillCondition: make([]float32, batch.Frames*h), DiagonalTarget: make([]float32, batch.Frames*c), DistillTarget: make([]float32, batch.Frames*c), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, 4*batch.Frames)}
	}
	for row, active := range samples.Mask {
		if !active {
			continue
		}
		if privateForward != nil && privateForward.handoffLease != nil {
			privateForward.handoffRow = row
		}
		condition := forward.Z[row*h : (row+1)*h]
		target := batch.NormalizedLatents[row*c : (row+1)*c]
		diagNoise := samples.Noise[row*c : (row+1)*c]
		time := samples.DiagonalTime[row]
		xTime, desired := workspace.xTime, workspace.desired
		var prediction []float32
		var tape *flowHeadTape
		var primaryTape, ordinaryBackward *flowTapeScratch
		if reuseTapes {
			workspace.primaryFlowTape.reset()
			workspace.endpointFlowBackward.reset()
			primaryTape = workspace.primaryFlowTape
			ordinaryBackward = workspace.endpointFlowBackward
		}
		diagonalForward, forwardFused := flow.TrainingBackend.(TrainingDiagonalForwardBackend)
		forwardFused = forwardFused && diagonalForward.DiagonalForwardEnabled()
		if forwardFused {
			in := DiagonalForwardInputs{Condition: condition, Target: target, Noise: diagNoise, Time: time}
			var r DiagonalForwardResult
			var e error
			if admitted, ok := flow.TrainingBackend.(trainingDiagonalAdmissionBackend); ok && admitted.diagonalAdmissionEnabled() {
				r, e = admitted.evaluateDiagonalAdmitted(flow, in, logvars[row])
			} else {
				r, e = diagonalForward.EvaluateDiagonalForward(flow, in)
			}
			if e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			if e = validateDiagonalForwardResult(r, in); e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			xTime, desired, prediction = r.Input, r.Desired, r.Prediction
		} else {
			if flow.BranchBackend != nil {
				values, e := evaluateTrainingBranch(flow.BranchBackend, flow, "interpolate", [][]float32{target, diagNoise}, time)
				if e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				xTime, desired = values[0], values[1]
			} else {
				for i := 0; i < c; i++ {
					xTime[i] = time*target[i] + (1-time)*diagNoise[i]
					desired[i] = target[i] - diagNoise[i]
				}
			}
			if flow.TrainingBackend != nil {
				initial, err := evaluateTrainingFlow(flow.TrainingBackend, flow, condition, []float32{time, time}, xTime, -1, make([]float32, c), make([]float32, c), false, 0)
				if err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
				if err := validateFlowBackendResult(flow, initial); err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
				prediction = initial.Output
			} else {
				var err error
				tape, prediction, err = flow.forwardTrainingScratch(condition, []float32{time, time}, xTime, primaryTape)
				if err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
			}
		}
		var diagLoss, rawSquare float64
		diagScale := config.PEqual * invValid
		var dCondition, dXTime []float32
		diagonal, diagFused := flow.TrainingBackend.(TrainingDiagonalBackend)
		diagFused = diagFused && diagonal.DiagonalEnabled()
		if diagFused {
			if stepBackend == nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("fused diagonal requires step reduction")
			}
			result, e := diagonal.EvaluateDiagonal(flow, condition, time, xTime, prediction, desired, logvars[row], diagScale)
			if e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			if e = validateDiagonalResult(result, condition, xTime, prediction, desired); e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			copy(reduction.DiagonalCondition[row*h:(row+1)*h], result.DCondition)
			copy(reduction.DiagonalTarget[row*c:(row+1)*c], result.DTarget)
			reduction.DiagonalLogvar[row] = result.DLogVariance
			reduction.RowMetrics[4*row], reduction.RowMetrics[4*row+1] = float32(result.Loss), float32(result.RawSquare)
		} else {
			var dPrediction, dLogvar []float32
			if flow.LossBackend != nil {
				r, e := flow.LossBackend.EvaluateLSDLoss(flow, prediction, desired, logvars[row])
				if e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				if e := validateLSDLossBoundary(r, prediction, desired); e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				diagLoss, rawSquare, dPrediction, dLogvar = r.Loss, r.RawSquare, r.DPrediction, []float32{r.DLogVariance}
			} else {
				diagLoss, dPrediction, dLogvar, err = LSDDiagonalLossAndGradient(prediction, desired, []float32{logvars[row]}, c, true)
				if err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
				for i := range prediction {
					difference := prediction[i] - desired[i]
					rawSquare += float64(difference) * float64(difference)
				}
			}
			if flow.BranchBackend != nil {
				values, e := evaluateTrainingBranch(flow.BranchBackend, flow, "scale", [][]float32{dPrediction}, diagScale)
				if e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				dPrediction = values[0]
			} else {
				for i := range dPrediction {
					dPrediction[i] *= diagScale
				}
			}
			if flow.TrainingBackend != nil {
				backward, err := evaluateTrainingFlow(flow.TrainingBackend, flow, condition, []float32{time, time}, xTime, -1, dPrediction, make([]float32, c), true, 1)
				if err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
				if err := validateFlowBackendResult(flow, backward); err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
				if _, resident := flow.TrainingBackend.(TrainingFlowAccumulator); !resident {
					g, err := flowGradientsFromNamed(flow, backward.Parameters)
					if err != nil {
						return TrainingStepMetrics{}, TrainingStepGradients{}, err
					}
					accumulateFlowTrainingGradients(flowGradients, g, 1)
				}
				dCondition, dXTime = backward.DCondition, backward.DInput
			} else {
				var err error
				dCondition, _, dXTime, err = flow.backwardTrainingScratch(tape, dPrediction, flowGradients, ordinaryBackward)
				if err != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, err
				}
			}
			if stepBackend != nil {
				copy(reduction.DiagonalCondition[row*h:(row+1)*h], dCondition)
			} else {
				for i := range dCondition {
					dZ[row*h+i] += dCondition[i]
				}
			}
			if flow.BranchBackend != nil {
				values, e := evaluateTrainingBranch(flow.BranchBackend, flow, "diagonal-target", [][]float32{dXTime, dPrediction}, time)
				if e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				if stepBackend != nil {
					copy(reduction.DiagonalTarget[row*c:(row+1)*c], values[0])
				} else {
					addInPlace(directTarget[row*c:(row+1)*c], values[0])
				}
			} else {
				for i := 0; i < c; i++ {
					directTarget[row*c+i] += time*dXTime[i] - dPrediction[i]
				}
			}
			if stepBackend != nil {
				reduction.DiagonalLogvar[row] = dLogvar[0]
				reduction.RowMetrics[4*row], reduction.RowMetrics[4*row+1] = float32(diagLoss), float32(rawSquare)
			} else {
				dDiagLogvar[row] = dLogvar[0] * diagScale
				diagonalObjective += diagLoss / float64(valid)
				metrics.FlowDiagonal += rawSquare / float64(valid)
			}

		}
		distillScale := (1 - config.PEqual) * invValid
		var endpointTape, primaryBackward, endpointBackward *flowTapeScratch
		if reuseTapes {
			workspace.primaryFlowTape.reset()
			workspace.endpointFlowTape.reset()
			workspace.primaryFlowBackward.reset()
			workspace.endpointFlowBackward.reset()
			primaryTape, endpointTape, primaryBackward, endpointBackward = workspace.primaryFlowTape, workspace.endpointFlowTape, workspace.primaryFlowBackward, workspace.endpointFlowBackward
		}
		var distill LSDDistillResult
		var distillGradients LSDDistillGradients
		if flow.TrainingBackend != nil {
			var err error
			objectiveScaled := false
			if native, ok := flow.TrainingBackend.(*ResidentTrainingBackend); ok {
				objectiveScaled = native.DistillEnabled()
			}
			distill, distillGradients, err = flow.lsdDistillTrainingStepBackendScaled(flow.TrainingBackend, condition, samples.DistillS[row], samples.DistillT[row], samples.Noise[row*c:(row+1)*c], target, logvars[batch.Frames+row], distillScale)
			if err != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, err
			}
			if _, resident := flow.TrainingBackend.(TrainingFlowAccumulator); !resident {
				accumulateFlowTrainingGradients(flowGradients, distillGradients.Flow, distillScale)
			}
			if stepBackend != nil {
				reduction.DistillObjectiveScaled = objectiveScaled
			} else if objectiveScaled {
				// Input adjoints already carry the objective; logvar remains raw.
				distillGradients.DLogVariance *= distillScale
			} else if flow.BranchBackend != nil {
				for _, x := range []*[]float32{&distillGradients.Condition, &distillGradients.Target} {
					values, e := evaluateTrainingBranch(flow.BranchBackend, flow, "scale", [][]float32{*x}, distillScale)
					if e != nil {
						return TrainingStepMetrics{}, TrainingStepGradients{}, e
					}
					*x = values[0]
				}
				values, e := evaluateTrainingBranch(flow.BranchBackend, flow, "scale", [][]float32{{distillGradients.DLogVariance}}, distillScale)
				if e != nil {
					return TrainingStepMetrics{}, TrainingStepGradients{}, e
				}
				distillGradients.DLogVariance = values[0][0]
			} else {
				for i := range distillGradients.Condition {
					distillGradients.Condition[i] *= distillScale
				}
				for i := range distillGradients.Target {
					distillGradients.Target[i] *= distillScale
				}
				distillGradients.DLogVariance *= distillScale
			}
		} else {
			var err error
			distill, distillGradients, err = flow.lsdDistillForwardBackwardScratch(condition, samples.DistillS[row], samples.DistillT[row], samples.Noise[row*c:(row+1)*c], target, logvars[batch.Frames+row], flowGradients, discardFlowGradients, distillScale, primaryTape, endpointTape, primaryBackward, endpointBackward)
			if err != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, err
			}
		}
		if stepBackend != nil {
			copy(reduction.DistillCondition[row*h:(row+1)*h], distillGradients.Condition)
			copy(reduction.DistillTarget[row*c:(row+1)*c], distillGradients.Target)
			reduction.DistillLogvar[row] = distillGradients.DLogVariance
			reduction.RowMetrics[4*row+2], reduction.RowMetrics[4*row+3] = float32(distill.Loss), float32(distill.RawSquare)
		} else {
			for i := range distillGradients.Condition {
				dZ[row*h+i] += distillGradients.Condition[i]
			}
			for i := 0; i < c; i++ {
				directTarget[row*c+i] += distillGradients.Target[i]
			}
			dDistillLogvar[row] = distillGradients.DLogVariance
			distillObjective += distill.Loss / float64(valid)
			metrics.FlowDistill += distill.RawSquare / float64(valid)
		}
	}
	if privateForward != nil && privateForward.handoffLease != nil {
		if err := privateForward.closeFlowHandoff(); err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
	}
	if combinedWeight && stepBackend == nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step weighting requires reduction owner")
	}
	if stepBackend != nil {
		var r StepReductionResult
		var e error
		if combinedBackbone {
			if !combinedWeight || !combineEOS || flowLMTape.backbone == nil || flowLMTape.eosBackbone == nil || !zeroTrainingSeed(dEOS) || len(flowLMTape.eosMask) != batch.Frames || flowLMTape.eosWeight != config.EOSLossWeight || !stepBackbone.stepBackboneEnabled() {
				return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step backbone changed/invalid captured EOS")
			}
			var result stepBackboneTrainingResult
			result, e = stepBackbone.evaluateStepBackbone(flowLM, flow, weighting, batch, forward.Z, reduction, config, WeightSampleInputs{samples.DiagonalTime, samples.DistillS, samples.DistillT, reduction.DiagonalLogvar, reduction.DistillLogvar})
			r, combinedInputs = result.Reduction, result.Inputs
		} else if combinedWeight {
			if !stepWeight.stepWeightEnabled() {
				return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step weighting changed during step")
			}
			r, e = stepWeight.evaluateStepWeight(flow, weighting, reduction, config, WeightSampleInputs{samples.DiagonalTime, samples.DistillS, samples.DistillT, reduction.DiagonalLogvar, reduction.DistillLogvar})
		} else {
			r, e = stepBackend.EvaluateStepReduction(flow, reduction, config)
		}
		if e != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, e
		}
		if reducedResults {
			if !stepResultPolicy.stepResultsEnabled() {
				return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step result policy changed")
			}
		} else {
			if e := validateStepReductionResult(r, reduction); e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			dZ, directTarget = r.DCondition, r.DTarget
		}
		dDiagLogvar, dDistillLogvar, metrics = r.DiagonalLogvar, r.DistillLogvar, r.Metrics
	}
	dLogvars := workspace.dLogvars
	if !packed {
		copy(dLogvars, dDiagLogvar)
		copy(dLogvars[len(dDiagLogvar):], dDistillLogvar)
	}
	var weightingGradients *LSDWeightGradients
	if weightBackend != nil {
		if packed && !sampleBackend.SampleWeightEnabled() {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("weighting backend changed during step")
		}
		if combinedWeight {
			// Weighting gradients accumulated inside the admitted reduction tape.
		} else if rootsOnly {
			if !weightRoots.weightRootsEnabled() {
				return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("weight roots changed during step")
			}
			if e := weightRoots.accumulateSampleWeight(weighting, WeightSampleInputs{samples.DiagonalTime, samples.DistillS, samples.DistillT, dDiagLogvar, dDistillLogvar}); e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
		} else {
			var r WeightTrainingResult
			var e error
			if packed {
				r, e = sampleBackend.EvaluateSampleWeight(weighting, WeightSampleInputs{samples.DiagonalTime, samples.DistillS, samples.DistillT, dDiagLogvar, dDistillLogvar}, true)
			} else {
				r, e = weightBackend.EvaluateWeight(weighting, weightInputs, dLogvars, weightRows, true)
			}
			if e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
			if e := validateWeightBoundary(r, weightRows, weightInputs, dLogvars, samples.DiagonalTime, samples.DistillS, samples.DistillT, dDiagLogvar, dDistillLogvar); e != nil {
				return TrainingStepMetrics{}, TrainingStepGradients{}, e
			}
		}
		weightingGradients = &LSDWeightGradients{Layers: make([]LinearF32Gradient, len(weighting.Layers))}
		for i, l := range weighting.Layers {
			weightingGradients.Layers[i] = newLinearGradient(l)
		}
	} else {
		if weighting.TrainingBackend != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("weighting backend injected after forward")
		}
		weightingGradients, err = weighting.backward(weightTape, dLogvars, weightRows)
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
	}
	flowLMGradients, inputGradients := workspace.flowLMGradients, workspace.flowLMInputGradients
	if combinedBackbone {
		copy(inputGradients.VoiceLatents, combinedInputs.VoiceLatents)
		copy(inputGradients.NormalizedLatents, combinedInputs.NormalizedLatents)
	} else {
		flowLMGradients, inputGradients, err = flowLM.backwardTrainingTapeInto(flowLMTape, batch, dZ, dEOS, workspace.flowLMGradients, workspace.flowLMInputGradients)
	}
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	if combinedBackbone {
		// Final direct-target addition already executed inside the composed tape.
	} else if stepBackend != nil {
		result, e := stepBackend.EvaluateStepInputGradient(flow, inputGradients.NormalizedLatents, directTarget)
		if e != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, e
		}
		if len(result) != len(directTarget) || !finiteF32(result) || f32SlicesOverlap(result, inputGradients.NormalizedLatents) || f32SlicesOverlap(result, directTarget) {
			return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("step input gradient boundary invalid")
		}
		copy(inputGradients.NormalizedLatents, result)
	} else {
		addInPlace(inputGradients.NormalizedLatents, directTarget)
		metrics.FlowLoss = float64(config.PEqual)*diagonalObjective + float64(1-config.PEqual)*distillObjective
		metrics.Loss = metrics.FlowLoss + float64(config.EOSLossWeight)*metrics.EOS
	}
	return metrics, TrainingStepGradients{FlowLM: flowLMGradients, Flow: flowGradients, Inputs: inputGradients, DiagonalLogVariance: dDiagLogvar, DistillLogVariance: dDistillLogvar, Weighting: weightingGradients}, nil
}

func validateTrainingStepSamples(batch FlowLMTrainingBatch, samples TrainingStepSamples, hidden, latent int) error {
	_ = hidden
	noiseElements, ok := checked.MulInt(batch.Frames, latent)
	if !ok || len(samples.Noise) != noiseElements || len(samples.DiagonalTime) != batch.Frames || len(samples.DistillS) != batch.Frames || len(samples.DistillT) != batch.Frames {
		return fmt.Errorf("invalid Pocket TTS training sample shape")
	}
	seenPadding := false
	for row, active := range samples.Mask {
		if !active {
			seenPadding = true
		} else if seenPadding {
			return fmt.Errorf("Pocket TTS training mask is not a contiguous prefix")
		}
		for _, value := range []float32{samples.DiagonalTime[row], samples.DistillS[row], samples.DistillT[row]} {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS training sample is non-finite")
			}
		}
		if active && (samples.DiagonalTime[row] < 0 || samples.DiagonalTime[row] > 1 || samples.DistillS[row] < 0 || samples.DistillT[row] > 1 || samples.DistillS[row] > samples.DistillT[row]) {
			return fmt.Errorf("Pocket TTS training sample time is invalid")
		}
	}
	for _, values := range [][]float32{samples.Noise} {
		for _, value := range values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS training noise is non-finite")
			}
		}
	}
	return nil
}

func addFlowHeadGradients(dst, src *FlowHeadGradients, scale float32) {
	addLinearGradient(&dst.Input, src.Input, scale)
	addLinearGradient(&dst.Condition, src.Condition, scale)
	for i := range dst.Time {
		addLinearGradient(&dst.Time[i].FC1, src.Time[i].FC1, scale)
		addLinearGradient(&dst.Time[i].FC2, src.Time[i].FC2, scale)
		addScaled(dst.Time[i].RMSWeight, src.Time[i].RMSWeight, scale)
	}
	for i := range dst.Blocks {
		addScaled(dst.Blocks[i].NormWeight, src.Blocks[i].NormWeight, scale)
		addScaled(dst.Blocks[i].NormBias, src.Blocks[i].NormBias, scale)
		addLinearGradient(&dst.Blocks[i].FC1, src.Blocks[i].FC1, scale)
		addLinearGradient(&dst.Blocks[i].FC2, src.Blocks[i].FC2, scale)
		addLinearGradient(&dst.Blocks[i].Modulation, src.Blocks[i].Modulation, scale)
	}
	addLinearGradient(&dst.Final.Linear, src.Final.Linear, scale)
	addLinearGradient(&dst.Final.Modulation, src.Final.Modulation, scale)
}
func addLinearGradient(dst *LinearF32Gradient, src LinearF32Gradient, scale float32) {
	addScaled(dst.Weight, src.Weight, scale)
	addScaled(dst.Bias, src.Bias, scale)
}
func addScaled(dst, src []float32, scale float32) {
	for i, value := range src {
		dst[i] += scale * value
	}
}
