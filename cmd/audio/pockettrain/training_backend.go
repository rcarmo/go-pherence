//go:build linux

package main

import (
	"context"
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

// Error retains owners of uncertain native resources. The top-level CLI exits
// on failure; an embedding caller must keep this error until explicit recovery.
type trainingResourcesRetainedError struct {
	err   error
	owner any
}

func (e *trainingResourcesRetainedError) Error() string {
	return fmt.Sprintf("training resources retained: %v", e.err)
}
func (e *trainingResourcesRetainedError) Unwrap() error { return e.err }

func trainingBackendScope(name string, optin bool, devicePin string, maxBytes int64) (string, error) {
	if name == "simd" {
		return "simd", nil
	}
	if name == "vulkan-resident-sharedworkspace-hybrid" {
		if !optin || !validDigest(devicePin) || maxBytes <= 0 || maxBytes > 2<<30 {
			return "", fmt.Errorf("sharedworkspace requires device opt-in, runtime pin and unchanged byte ceiling")
		}
		return name + "-v1", nil
	}
	if name != "ptx-attention-hybrid" && name != "vulkan-attention-hybrid" && name != "ptx-flow-hybrid" && name != "vulkan-flow-hybrid" && name != "ptx-transformer-hybrid" && name != "vulkan-transformer-hybrid" && name != "ptx-optimizer-hybrid" && name != "vulkan-optimizer-hybrid" && name != "ptx-resident-hybrid" && name != "vulkan-resident-hybrid" && name != "ptx-resident-weight-hybrid" && name != "vulkan-resident-weight-hybrid" && name != "ptx-resident-projection-hybrid" && name != "vulkan-resident-projection-hybrid" && name != "ptx-resident-sequence-hybrid" && name != "vulkan-resident-sequence-hybrid" && name != "ptx-resident-time-hybrid" && name != "vulkan-resident-time-hybrid" && name != "ptx-resident-eos-hybrid" && name != "vulkan-resident-eos-hybrid" && name != "ptx-resident-loss-hybrid" && name != "vulkan-resident-loss-hybrid" && name != "ptx-resident-branch-hybrid" && name != "vulkan-resident-branch-hybrid" && name != "ptx-resident-step-hybrid" && name != "vulkan-resident-step-hybrid" && name != "ptx-resident-pruned-hybrid" && name != "vulkan-resident-pruned-hybrid" && name != "ptx-resident-globalnorm-hybrid" && name != "ptx-resident-devicescale-hybrid" && name != "ptx-resident-sampleweight-hybrid" && name != "ptx-resident-lifetime-hybrid" && name != "ptx-resident-statecheck-hybrid" && name != "ptx-resident-zeroconst-hybrid" && name != "ptx-resident-eosfused-hybrid" && name != "ptx-resident-audiofused-hybrid" && name != "ptx-resident-voicefused-hybrid" && name != "ptx-resident-conditioner-hybrid" && name != "ptx-resident-backbone-hybrid" && name != "ptx-resident-eosbackbone-hybrid" && name != "ptx-resident-diagfused-hybrid" && name != "ptx-resident-endpointfused-hybrid" && name != "ptx-resident-primaryfused-hybrid" && name != "ptx-resident-distillfused-hybrid" && name != "ptx-resident-forwardfused-hybrid" && name != "ptx-resident-diagforward-hybrid" && name != "ptx-resident-weightroots-hybrid" && name != "ptx-resident-stepweight-hybrid" && name != "ptx-resident-stepbackbone-hybrid" && name != "ptx-resident-stepresults-hybrid" && name != "ptx-resident-distillresults-hybrid" && name != "ptx-resident-eosforward-hybrid" && name != "ptx-resident-forwardresults-hybrid" && name != "ptx-resident-embeddingindex-hybrid" && name != "ptx-resident-embeddingcompact-hybrid" && name != "ptx-resident-diagadmission-hybrid" && name != "ptx-resident-distilladmission-hybrid" && name != "ptx-resident-flowhandoff-hybrid" && name != "ptx-resident-eosadmission-hybrid" && name != "ptx-resident-devicehandoff-hybrid" && name != "ptx-resident-predictionhandoff-hybrid" && name != "vulkan-resident-globalnorm-hybrid" && name != "vulkan-resident-devicescale-hybrid" && name != "vulkan-resident-sampleweight-hybrid" && name != "vulkan-resident-lifetime-hybrid" && name != "vulkan-resident-statecheck-hybrid" && name != "vulkan-resident-zeroconst-hybrid" && name != "vulkan-resident-eosfused-hybrid" && name != "vulkan-resident-audiofused-hybrid" && name != "vulkan-resident-voicefused-hybrid" && name != "vulkan-resident-conditioner-hybrid" && name != "vulkan-resident-backbone-hybrid" && name != "vulkan-resident-eosbackbone-hybrid" && name != "vulkan-resident-diagfused-hybrid" && name != "vulkan-resident-endpointfused-hybrid" && name != "vulkan-resident-primaryfused-hybrid" && name != "vulkan-resident-distillfused-hybrid" && name != "vulkan-resident-forwardfused-hybrid" && name != "vulkan-resident-diagforward-hybrid" && name != "vulkan-resident-weightroots-hybrid" && name != "vulkan-resident-stepweight-hybrid" && name != "vulkan-resident-stepbackbone-hybrid" && name != "vulkan-resident-stepresults-hybrid" && name != "vulkan-resident-distillresults-hybrid" && name != "vulkan-resident-eosforward-hybrid" && name != "vulkan-resident-forwardresults-hybrid" && name != "vulkan-resident-embeddingindex-hybrid" && name != "vulkan-resident-embeddingcompact-hybrid" && name != "vulkan-resident-diagadmission-hybrid" && name != "vulkan-resident-distilladmission-hybrid" && name != "vulkan-resident-flowhandoff-hybrid" && name != "vulkan-resident-eosadmission-hybrid" && name != "vulkan-resident-devicehandoff-hybrid" && name != "vulkan-resident-predictionhandoff-hybrid" {
		return "", fmt.Errorf("complete Vulkan/PTX loss/backward integration is not implemented")
	}
	if !optin || !validDigest(devicePin) || maxBytes <= 0 || maxBytes > 2<<30 {
		return "", fmt.Errorf("hybrid backend requires explicit device opt-in, device/runtime SHA256 identity and byte ceiling")
	}
	return name + "-v1", nil
}

// trainingBackendOwners binds one explicit device scope. Combining attention and
// flow scopes is not admitted. Keep this owner reachable if cleanup fails.
type trainingBackendOwners struct {
	attention   pockettts.TrainingAttentionBackend
	flow        pockettts.TrainingFlowBackend
	transformer pockettts.TrainingTransformerBackend
	optimizer   pockettts.TrainingOptimizerBackend
	weight      pockettts.TrainingWeightBackend
	projection  pockettts.TrainingProjectionBackend
	sequence    pockettts.TrainingSequenceBackend
	eos         pockettts.TrainingEOSBackend
	loss        pockettts.TrainingLSDLossBackend
	branch      pockettts.TrainingBranchBackend
	step        pockettts.TrainingStepReductionBackend
}

func openTrainingBackend(ctx context.Context, scope string, maxBytes int64) (trainingBackendOwners, func() error, error) {
	var owners trainingBackendOwners
	if ctx == nil {
		return owners, nil, fmt.Errorf("nil training context")
	}
	if err := ctx.Err(); err != nil {
		return owners, nil, err
	}
	if scope != "simd" && (maxBytes <= 0 || maxBytes > 2<<30) {
		return owners, nil, fmt.Errorf("invalid device tensor byte ceiling")
	}
	switch scope {
	case "vulkan-resident-sharedworkspace-hybrid-v1":
		return openSharedWorkspaceBackend(ctx, maxBytes)
	case "ptx-flow-hybrid-v1", "ptx-transformer-hybrid-v1", "ptx-optimizer-hybrid-v1", "ptx-resident-hybrid-v1", "ptx-resident-weight-hybrid-v1", "ptx-resident-projection-hybrid-v1", "ptx-resident-sequence-hybrid-v1", "ptx-resident-time-hybrid-v1", "ptx-resident-eos-hybrid-v1", "ptx-resident-loss-hybrid-v1", "ptx-resident-branch-hybrid-v1", "ptx-resident-step-hybrid-v1", "ptx-resident-pruned-hybrid-v1", "ptx-resident-globalnorm-hybrid-v1", "ptx-resident-devicescale-hybrid-v1", "ptx-resident-sampleweight-hybrid-v1", "ptx-resident-lifetime-hybrid-v1", "ptx-resident-statecheck-hybrid-v1", "ptx-resident-zeroconst-hybrid-v1", "ptx-resident-eosfused-hybrid-v1", "ptx-resident-audiofused-hybrid-v1", "ptx-resident-voicefused-hybrid-v1", "ptx-resident-conditioner-hybrid-v1", "ptx-resident-backbone-hybrid-v1", "ptx-resident-eosbackbone-hybrid-v1", "ptx-resident-diagfused-hybrid-v1", "ptx-resident-endpointfused-hybrid-v1", "ptx-resident-primaryfused-hybrid-v1", "ptx-resident-distillfused-hybrid-v1", "ptx-resident-forwardfused-hybrid-v1", "ptx-resident-diagforward-hybrid-v1", "ptx-resident-weightroots-hybrid-v1", "ptx-resident-stepweight-hybrid-v1", "ptx-resident-stepbackbone-hybrid-v1", "ptx-resident-stepresults-hybrid-v1", "ptx-resident-distillresults-hybrid-v1", "ptx-resident-eosforward-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1", "ptx-resident-embeddingindex-hybrid-v1", "ptx-resident-embeddingcompact-hybrid-v1", "ptx-resident-diagadmission-hybrid-v1", "ptx-resident-distilladmission-hybrid-v1", "ptx-resident-flowhandoff-hybrid-v1", "ptx-resident-eosadmission-hybrid-v1", "ptx-resident-devicehandoff-hybrid-v1", "ptx-resident-predictionhandoff-hybrid-v1":
		op, err := nvidia.NewTrainingF32()
		if err != nil {
			if op != nil {
				return owners, op.Close, err
			}
			return owners, nil, err
		}
		if scope == "ptx-resident-hybrid-v1" || scope == "ptx-resident-weight-hybrid-v1" || scope == "ptx-resident-projection-hybrid-v1" || scope == "ptx-resident-sequence-hybrid-v1" || scope == "ptx-resident-time-hybrid-v1" || scope == "ptx-resident-eos-hybrid-v1" || scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
			backend := &pockettts.ResidentTrainingBackend{Driver: &nvidia.PTXResidentDriver{Operator: op}, Scope: scope, MaxResidentBytes: maxBytes, MaxTapeBytes: maxBytes}
			owners.flow, owners.transformer, owners.optimizer = backend, backend, backend
			if scope == "ptx-resident-weight-hybrid-v1" || scope == "ptx-resident-projection-hybrid-v1" || scope == "ptx-resident-sequence-hybrid-v1" || scope == "ptx-resident-time-hybrid-v1" || scope == "ptx-resident-eos-hybrid-v1" || scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.weight = backend
			}
			if scope == "ptx-resident-projection-hybrid-v1" || scope == "ptx-resident-sequence-hybrid-v1" || scope == "ptx-resident-time-hybrid-v1" || scope == "ptx-resident-eos-hybrid-v1" || scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.projection = backend
			}
			if scope == "ptx-resident-sequence-hybrid-v1" || scope == "ptx-resident-time-hybrid-v1" || scope == "ptx-resident-eos-hybrid-v1" || scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.sequence = backend
			}
			if scope == "ptx-resident-eos-hybrid-v1" || scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.eos = backend
			}
			if scope == "ptx-resident-loss-hybrid-v1" || scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.loss = backend
			}
			if scope == "ptx-resident-branch-hybrid-v1" || scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.branch = backend
			}
			if scope == "ptx-resident-step-hybrid-v1" || scope == "ptx-resident-pruned-hybrid-v1" || scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "ptx-resident-devicescale-hybrid-v1" || scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "ptx-resident-lifetime-hybrid-v1" || scope == "ptx-resident-statecheck-hybrid-v1" || scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "ptx-resident-eosfused-hybrid-v1" || scope == "ptx-resident-audiofused-hybrid-v1" || scope == "ptx-resident-voicefused-hybrid-v1" || scope == "ptx-resident-conditioner-hybrid-v1" || scope == "ptx-resident-backbone-hybrid-v1" || scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "ptx-resident-diagfused-hybrid-v1" || scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "ptx-resident-distillfused-hybrid-v1" || scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "ptx-resident-diagforward-hybrid-v1" || scope == "ptx-resident-weightroots-hybrid-v1" || scope == "ptx-resident-stepweight-hybrid-v1" || scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.step = backend
			}
			return owners, func() error {
				if err := backend.Close(); err != nil {
					return err
				}
				return op.Close()
			}, nil
		}
		if scope == "ptx-optimizer-hybrid-v1" {
			backend := &pockettts.ResidentOptimizerBackend{Driver: &nvidia.PTXResidentDriver{Operator: op}, Scope: scope, MaxBytes: maxBytes}
			owners.optimizer = backend
			return owners, func() error {
				if err := backend.Close(); err != nil {
					return err
				}
				return op.Close()
			}, nil
		}
		executor := &pockettts.PTXFlowTapeExecutor{Context: ctx, Operator: op, MaxBytes: maxBytes}
		if scope == "ptx-transformer-hybrid-v1" {
			backend := &pockettts.TransformerTapeTrainingBackend{Executor: executor, MaxBytes: maxBytes}
			owners.transformer = backend
			return owners, backend.Close, nil
		}
		backend := &pockettts.FlowTapeTrainingBackend{Executor: executor, MaxBytes: maxBytes}
		owners.flow = backend
		return owners, backend.Close, nil
	case "vulkan-flow-hybrid-v1", "vulkan-transformer-hybrid-v1", "vulkan-optimizer-hybrid-v1", "vulkan-resident-hybrid-v1", "vulkan-resident-weight-hybrid-v1", "vulkan-resident-projection-hybrid-v1", "vulkan-resident-sequence-hybrid-v1", "vulkan-resident-time-hybrid-v1", "vulkan-resident-eos-hybrid-v1", "vulkan-resident-loss-hybrid-v1", "vulkan-resident-branch-hybrid-v1", "vulkan-resident-step-hybrid-v1", "vulkan-resident-pruned-hybrid-v1", "vulkan-resident-globalnorm-hybrid-v1", "vulkan-resident-devicescale-hybrid-v1", "vulkan-resident-sampleweight-hybrid-v1", "vulkan-resident-lifetime-hybrid-v1", "vulkan-resident-statecheck-hybrid-v1", "vulkan-resident-zeroconst-hybrid-v1", "vulkan-resident-eosfused-hybrid-v1", "vulkan-resident-audiofused-hybrid-v1", "vulkan-resident-voicefused-hybrid-v1", "vulkan-resident-conditioner-hybrid-v1", "vulkan-resident-backbone-hybrid-v1", "vulkan-resident-eosbackbone-hybrid-v1", "vulkan-resident-diagfused-hybrid-v1", "vulkan-resident-endpointfused-hybrid-v1", "vulkan-resident-primaryfused-hybrid-v1", "vulkan-resident-distillfused-hybrid-v1", "vulkan-resident-forwardfused-hybrid-v1", "vulkan-resident-diagforward-hybrid-v1", "vulkan-resident-weightroots-hybrid-v1", "vulkan-resident-stepweight-hybrid-v1", "vulkan-resident-stepbackbone-hybrid-v1", "vulkan-resident-stepresults-hybrid-v1", "vulkan-resident-distillresults-hybrid-v1", "vulkan-resident-eosforward-hybrid-v1", "vulkan-resident-forwardresults-hybrid-v1", "vulkan-resident-embeddingindex-hybrid-v1", "vulkan-resident-embeddingcompact-hybrid-v1", "vulkan-resident-diagadmission-hybrid-v1", "vulkan-resident-distilladmission-hybrid-v1", "vulkan-resident-flowhandoff-hybrid-v1", "vulkan-resident-eosadmission-hybrid-v1", "vulkan-resident-devicehandoff-hybrid-v1", "vulkan-resident-predictionhandoff-hybrid-v1":
		if !vulkan.VulkanInit() {
			return owners, nil, fmt.Errorf("Vulkan unavailable")
		}
		op, err := vulkan.NewVkTrainingF32(ctx)
		if err != nil {
			if op != nil {
				return owners, op.Close, err
			}
			return owners, nil, err
		}
		if scope == "vulkan-resident-hybrid-v1" || scope == "vulkan-resident-weight-hybrid-v1" || scope == "vulkan-resident-projection-hybrid-v1" || scope == "vulkan-resident-sequence-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
			backend := &pockettts.ResidentTrainingBackend{Driver: &vulkan.VulkanResidentDriver{Operator: op}, Scope: scope, MaxResidentBytes: maxBytes, MaxTapeBytes: maxBytes}
			owners.flow, owners.transformer, owners.optimizer = backend, backend, backend
			if scope == "vulkan-resident-weight-hybrid-v1" || scope == "vulkan-resident-projection-hybrid-v1" || scope == "vulkan-resident-sequence-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.weight = backend
			}
			if scope == "vulkan-resident-projection-hybrid-v1" || scope == "vulkan-resident-sequence-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.projection = backend
			}
			if scope == "vulkan-resident-sequence-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.sequence = backend
			}
			if scope == "vulkan-resident-eos-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.eos = backend
			}
			if scope == "vulkan-resident-loss-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.loss = backend
			}
			if scope == "vulkan-resident-branch-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.branch = backend
			}
			if scope == "vulkan-resident-step-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
				owners.step = backend
			}
			return owners, func() error {
				if err := backend.Close(); err != nil {
					return err
				}
				return op.Close()
			}, nil
		}
		if scope == "vulkan-optimizer-hybrid-v1" {
			driver := &vulkan.VulkanResidentDriver{Operator: op}
			backend := &pockettts.ResidentOptimizerBackend{Driver: driver, Scope: scope, MaxBytes: maxBytes}
			owners.optimizer = backend
			return owners, func() error {
				if err := backend.Close(); err != nil {
					return err
				}
				if err := driver.Close(); err != nil {
					return err
				}
				return op.Close()
			}, nil
		}
		executor := &pockettts.VulkanFlowTapeExecutor{Context: ctx, Operator: op, MaxBytes: maxBytes}
		if scope == "vulkan-transformer-hybrid-v1" {
			backend := &pockettts.TransformerTapeTrainingBackend{Executor: executor, MaxBytes: maxBytes}
			owners.transformer = backend
			return owners, backend.Close, nil
		}
		backend := &pockettts.FlowTapeTrainingBackend{Executor: executor, MaxBytes: maxBytes}
		owners.flow = backend
		return owners, backend.Close, nil
	default:
		attention, cleanup, err := openTrainingAttentionBackend(ctx, scope, maxBytes)
		owners.attention = attention
		return owners, cleanup, err
	}
}

// Keep the attention-hybrid v1 digest unchanged for existing bundles. Flow
// hybrids have a distinct scope; all device scopes bind the runtime pin.
func trainingBackendOptionsHash(base, scope, pin string) string {
	if scope == "simd" {
		return base
	}
	if scope == "vulkan-resident-sharedworkspace-hybrid-v1" {
		return hashText(base + ";resident-sharedworkspace-backend=" + scope + ";device-runtime=" + pin)
	}
	kind := "attention-backend"
	if scope == "ptx-flow-hybrid-v1" || scope == "vulkan-flow-hybrid-v1" {
		kind = "flow-backend"
	}
	if scope == "ptx-transformer-hybrid-v1" || scope == "vulkan-transformer-hybrid-v1" {
		kind = "transformer-backend"
	}
	if scope == "ptx-optimizer-hybrid-v1" || scope == "vulkan-optimizer-hybrid-v1" {
		kind = "optimizer-backend"
	}
	if scope == "ptx-resident-hybrid-v1" || scope == "vulkan-resident-hybrid-v1" {
		kind = "resident-backend"
	}
	if scope == "ptx-resident-weight-hybrid-v1" || scope == "vulkan-resident-weight-hybrid-v1" {
		kind = "resident-weight-backend"
	}
	if scope == "ptx-resident-projection-hybrid-v1" || scope == "vulkan-resident-projection-hybrid-v1" {
		kind = "resident-projection-backend"
	}
	if scope == "ptx-resident-sequence-hybrid-v1" || scope == "vulkan-resident-sequence-hybrid-v1" {
		kind = "resident-sequence-backend"
	}
	if scope == "ptx-resident-time-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" {
		kind = "resident-time-backend"
	}
	if scope == "ptx-resident-eos-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" {
		kind = "resident-eos-backend"
	}
	if scope == "ptx-resident-loss-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" {
		kind = "resident-loss-backend"
	}
	if scope == "ptx-resident-branch-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" {
		kind = "resident-branch-backend"
	}
	if scope == "ptx-resident-step-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" {
		kind = "resident-step-backend"
	}
	if scope == "ptx-resident-pruned-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" {
		kind = "resident-pruned-backend"
	}
	if scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" {
		kind = "resident-globalnorm-backend"
	}
	if scope == "ptx-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" {
		kind = "resident-devicescale-backend"
	}
	if scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" {
		kind = "resident-sampleweight-backend"
	}
	if scope == "ptx-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" {
		kind = "resident-lifetime-backend"
	}
	if scope == "ptx-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" {
		kind = "resident-statecheck-backend"
	}
	if scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" {
		kind = "resident-zeroconst-backend"
	}
	if scope == "ptx-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" {
		kind = "resident-eosfused-backend"
	}
	if scope == "ptx-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" {
		kind = "resident-audiofused-backend"
	}
	if scope == "ptx-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" {
		kind = "resident-voicefused-backend"
	}
	if scope == "ptx-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" {
		kind = "resident-conditioner-backend"
	}
	if scope == "ptx-resident-backbone-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" {
		kind = "resident-backbone-backend"
	}
	if scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" {
		kind = "resident-eosbackbone-backend"
	}
	if scope == "ptx-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" {
		kind = "resident-diagfused-backend"
	}
	if scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" {
		kind = "resident-endpointfused-backend"
	}
	if scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" {
		kind = "resident-primaryfused-backend"
	}
	if scope == "ptx-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" {
		kind = "resident-distillfused-backend"
	}
	if scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" {
		kind = "resident-forwardfused-backend"
	}
	if scope == "ptx-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" {
		kind = "resident-diagforward-backend"
	}
	if scope == "ptx-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" {
		kind = "resident-weightroots-backend"
	}
	if scope == "ptx-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" {
		kind = "resident-stepweight-backend"
	}
	if scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" {
		kind = "resident-stepbackbone-backend"
	}
	if scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
		kind = "resident-stepresults-backend"
	}
	if scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
		kind = "resident-distillresults-backend"
	}
	if scope == "ptx-resident-eosforward-hybrid-v1" || (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) || scope == "vulkan-resident-eosforward-hybrid-v1" || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
		kind = "resident-eosforward-backend"
	}
	if (scope == "ptx-resident-forwardresults-hybrid-v1" || (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))))) || (scope == "vulkan-resident-forwardresults-hybrid-v1" || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))))) {
		kind = "resident-forwardresults-backend"
	}
	if (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")))))) || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")))))) {
		kind = "resident-embeddingindex-backend"
	}
	if (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))) || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))) {
		kind = "resident-embeddingcompact-backend"
	}
	if (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")))) || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")))) {
		kind = "resident-diagadmission-backend"
	}
	if (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))) || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))) {
		kind = "resident-distilladmission-backend"
	}
	if (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")) || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")) {
		kind = "resident-flowhandoff-backend"
	}
	if (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1") || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1") {
		kind = "resident-eosadmission-backend"
	}
	if scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1" {
		kind = "resident-predictionhandoff-backend"
	}
	if scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" {
		kind = "resident-devicehandoff-backend"
	}
	return hashText(base + ";" + kind + "=" + scope + ";device-runtime=" + pin)
}

func trainingBackendNotice(scope string) string {
	if scope == "vulkan-resident-sharedworkspace-hybrid-v1" {
		return "guarded sharedworkspace hybrid: stable G, five persistent buffers per parameter, recomputed norm statistics; coherent mapped CPU copy-back, retained device handoffs and full snapshots; released-size qualification pending"
	}
	if scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" {
		return "retained-device handoff hybrid: coordinator-owned row condition and frozen forward children, shape-only backward sources and scalar guards; existing host forward downloads/overflow guards, step results and full snapshots retained; native qualification pending"
	}
	if scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1" {
		return "Resident prediction handoff hybrid: retained diagonal/distillation forward children provide read-only device condition/input/prediction/desired edges to backward tapes; original host guards/downloads and full snapshots retained. Conservative EOS admission may reject near-limit host-safe values. Native qualification pending."
	}
	if (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1") || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1") {
		return "Resident EOS admission hybrid: conservative integer absolute-dot flag before gradients, including padded rows; may reject near-limit host-safe inputs. Read-only transformer/flow handoff retained; original host hidden/EOS guard, prediction/backward downloads and full snapshots remain. Native execution qualification pending."
	}
	if (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")) || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")) {
		return "Resident forward handoff hybrid: transformer producer retained across flow row consumers with read-only device condition slices; host hidden/EOS overflow guards, prediction downloads, backward host boundaries and full snapshots retained. Native GPU execution/quality qualification pending."
	}
	if (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))) || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))) {
		return "experimental resident diagonal/distillation-admission hybrid: scalar read-only guards before row backward; prediction downloads and redundant CPU admission retained; compact embedding/device AdamW/EMA; handoffs/diagnostics/full snapshots/native qualification remain"
	}
	if (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")))) || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")))) {
		return "experimental resident diagonal-admission hybrid: scalar read-only prediction guard before backward; prediction downloads and redundant CPU admission retained; compact embedding/device AdamW/EMA; handoffs/diagnostics/full snapshots/native qualification remain"
	}
	if (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1"))))) || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1"))))) {
		return "experimental resident compact-embedding hybrid: compact token VJP without dense tape expansion; dense resident gradient/AdamW/EMA work retained; CPU handoffs/diagnostics/full snapshots remain; native qualification open"
	}
	if (scope == "ptx-resident-embeddingindex-hybrid-v1" || (scope == "ptx-resident-embeddingcompact-hybrid-v1" || (scope == "ptx-resident-diagadmission-hybrid-v1" || (scope == "ptx-resident-distilladmission-hybrid-v1" || (scope == "ptx-resident-flowhandoff-hybrid-v1" || (scope == "ptx-resident-eosadmission-hybrid-v1" || scope == "ptx-resident-devicehandoff-hybrid-v1" || scope == "ptx-resident-predictionhandoff-hybrid-v1")))))) || (scope == "vulkan-resident-embeddingindex-hybrid-v1" || (scope == "vulkan-resident-embeddingcompact-hybrid-v1" || (scope == "vulkan-resident-diagadmission-hybrid-v1" || (scope == "vulkan-resident-distilladmission-hybrid-v1" || (scope == "vulkan-resident-flowhandoff-hybrid-v1" || (scope == "vulkan-resident-eosadmission-hybrid-v1" || scope == "vulkan-resident-devicehandoff-hybrid-v1" || scope == "vulkan-resident-predictionhandoff-hybrid-v1")))))) {
		return "experimental resident indexed-embedding hybrid: sequence-order occurrence-index VJP with dense gradients; CPU hidden admission, flow handoffs/diagnostics/full snapshots retained; device AdamW/EMA; index setup/upload cost and native qualification open"
	}
	if scope == "ptx-resident-forwardresults-hybrid-v1" || scope == "vulkan-resident-forwardresults-hybrid-v1" {
		return "experimental resident selected-forward-result hybrid: private audio hidden/EOS scalar/finite witness; discarded sequence/prefix downloads omitted; public FlowLM outputs unchanged; CPU hidden overflow admission/device AdamW/EMA retained; flow handoffs, diagnostics and full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-eosforward-hybrid-v1" || scope == "ptx-resident-forwardresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || scope == "vulkan-resident-forwardresults-hybrid-v1" {
		return "experimental resident backbone/EOS forward hybrid: read-only combined transformer and EOS scalar; CPU hidden overflow admission gates all gradients; private selected results and device AdamW/EMA retained; flow handoffs, diagnostic sync and full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || scope == "ptx-resident-forwardresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || scope == "vulkan-resident-forwardresults-hybrid-v1" {
		return "experimental resident private distillation results hybrid: step-only condition/target/statistics plus finite noise/time witness; public LSD results unchanged; private final-step outputs and device AdamW/EMA retained; CPU prediction admission, diagnostic sync and full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-stepresults-hybrid-v1" || scope == "ptx-resident-distillresults-hybrid-v1" || scope == "ptx-resident-eosforward-hybrid-v1" || scope == "ptx-resident-forwardresults-hybrid-v1" || scope == "vulkan-resident-stepresults-hybrid-v1" || scope == "vulkan-resident-distillresults-hybrid-v1" || scope == "vulkan-resident-eosforward-hybrid-v1" || scope == "vulkan-resident-forwardresults-hybrid-v1" {
		return "experimental resident step-result hybrid: private final-step logvar/metrics/input downloads and direct-tail witness, discarded fullcondition/directtarget outputs omitted; public APIs intact; device AdamW/EMA retained; CPU prediction admission/forward flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-stepbackbone-hybrid-v1" || scope == "vulkan-resident-stepbackbone-hybrid-v1" {
		return "experimental resident step-backbone hybrid: reduction/weighting/EOS-backbone backward/final input gradient in one bound tape; public reduction values preserved; all parameters accumulate once; device AdamW/EMA retained; CPU prediction admission/forward flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-stepweight-hybrid-v1" || scope == "vulkan-resident-stepweight-hybrid-v1" {
		return "experimental resident step-weight hybrid: reduction scaled-seed edges feed sample-weight parameter roots in one tape; public reduction outputs preserved, accumulation once; device AdamW/EMA retained; CPU prediction admission/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-weightroots-hybrid-v1" || scope == "vulkan-resident-weightroots-hybrid-v1" {
		return "experimental resident weighting-roots hybrid: step-only sample-weight backward accumulates parameter roots without discarded output/input VJP downloads; public seeded API intact; device AdamW/EMA retained; CPU prediction admission/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-diagforward-hybrid-v1" || scope == "vulkan-resident-diagforward-hybrid-v1" {
		return "experimental resident diagonal-forward hybrid: interpolation/initial flow in one bound tape; three overflow/backward boundaries retained, no forward accumulation; device AdamW/EMA retained; CPU prediction admission/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-forwardfused-hybrid-v1" || scope == "vulkan-resident-forwardfused-hybrid-v1" {
		return "experimental resident distillation-forward hybrid: interpolation/primary JVP/advance/endpoint in one bound tape; six public/overflow-admission values retained, no forward accumulation; device AdamW/EMA retained; CPU prediction admission/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-distillfused-hybrid-v1" || scope == "vulkan-resident-distillfused-hybrid-v1" {
		return "experimental resident distillation-backward hybrid: endpoint input-only VJP and primary mixed/finish gradient edges in one bound tape; primary-only parameter accumulation scaled once; device AdamW/EMA retained; CPU forward prediction/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-primaryfused-hybrid-v1" || scope == "vulkan-resident-primaryfused-hybrid-v1" {
		return "experimental resident primary-finish hybrid: primary mixed backward and raw target/noise/time VJP in one bound tape; parameter objective scale once; endpoint input-only policy and device AdamW/EMA retained; CPU forward/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-endpointfused-hybrid-v1" || scope == "vulkan-resident-endpointfused-hybrid-v1" {
		return "experimental resident endpoint-loss hybrid: forward overflow admission retained; loss/endpoint input-only VJP/mixed primary seeds in bound tape; no endpoint parameter accumulation; device AdamW/EMA retained; CPU primary/distillation/transformer-flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-diagfused-hybrid-v1" || scope == "vulkan-resident-diagfused-hybrid-v1" {
		return "experimental resident diagonal-loss/flow hybrid: prediction overflow admission retained; loss seeds/scaling/flow VJP/directtarget in bound tape; device AdamW/EMA retained; CPU distillation/other flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-eosbackbone-hybrid-v1" || scope == "vulkan-resident-eosbackbone-hybrid-v1" {
		return "experimental resident EOS/backbone hybrid: EOS scalar forward loss retained; weighted EOS VJP and flow condition seed combine in backbone backward; device AdamW/EMA retained; CPU flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-backbone-hybrid-v1" || scope == "vulkan-resident-backbone-hybrid-v1" {
		return "experimental resident backbone hybrid: conditioner/transformer forward and VJP in one bound tape; public sequence retained, sequence-gradient host edge removed; device AdamW/EMA retained; CPU flow boundaries/diagnostic sync/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-conditioner-hybrid-v1" || scope == "vulkan-resident-conditioner-hybrid-v1" {
		return "experimental resident conditioner hybrid: raw voice/text/shifted-audio projection assembly VJP in bound tape;fused EOS/AdamW/EMA/state checks retained;CPU transformer/flow boundaries/full snapshots remain;native qualification open"
	}
	if scope == "ptx-resident-voicefused-hybrid-v1" || scope == "vulkan-resident-voicefused-hybrid-v1" {
		return "experimental resident fused-voice sequence hybrid: raw voice projection/sequence VJP in bound tape plus fused audio/EOS;device AdamW/EMA/state checks retained;CPU transformer/flow boundaries/full snapshots remain;native qualification open"
	}
	if scope == "ptx-resident-audiofused-hybrid-v1" || scope == "vulkan-resident-audiofused-hybrid-v1" {
		return "experimental resident fused-audio hybrid: BOS/shifted-input/affine VJP in bound tape plus fused EOS;zero/lifetime/statecheck/AdamW/EMA retained;CPU dispatch/other boundaries/full snapshots remain;native qualification open"
	}
	if scope == "ptx-resident-eosfused-hybrid-v1" || scope == "vulkan-resident-eosfused-hybrid-v1" {
		return "experimental resident fused-EOS hybrid: one bound EOS affine/BCE/VJP tape without logit-seed roundtrip; zero-constant/lifetime/statecheck/AdamW/EMA retained; CPU dispatch/boundaries/full snapshots remain;native qualification open"
	}
	if scope == "ptx-resident-zeroconst-hybrid-v1" || scope == "vulkan-resident-zeroconst-hybrid-v1" {
		return "experimental resident zero-constant hybrid: pinned read-only positive-zero device sources shared by extent,parameters excluded; lifetime/statecheck/AdamW/EMA retained; CPU dispatch/boundaries/diagnostics/full snapshots remain;native qualification open"
	}
	if scope == "ptx-resident-statecheck-hybrid-v1" || scope == "vulkan-resident-statecheck-hybrid-v1" {
		return "experimental resident state-check hybrid: device merged post-AdamW/EMA finite/nonnegative checks with one diagnostic download; lifetime/sample-weight tapes; CPU dispatch/norm diagnostics/boundaries/full snapshots remain; native qualification open"
	}
	if scope == "ptx-resident-lifetime-hybrid-v1" || scope == "vulkan-resident-lifetime-hybrid-v1" {
		return "experimental resident lifetime hybrid: pruned equal-size intermediate storage reuse and sample packing/device norm/AdamW/EMA; CPU dispatch/boundaries/diagnostics/full snapshots remain; native resource/performance qualification open"
	}
	if scope == "ptx-resident-sampleweight-hybrid-v1" || scope == "vulkan-resident-sampleweight-hybrid-v1" {
		return "experimental resident sample-weight hybrid: sample times/logvar seeds packed in weighting MLP tape; device norm/clip buffer AdamW/EMA and pruned component tapes; CPU dispatch/four-scalar diagnostic admission/post-update checks/full snapshots/inter-tape transfers remain"
	}
	if scope == "ptx-resident-devicescale-hybrid-v1" || scope == "vulkan-resident-devicescale-hybrid-v1" {
		return "experimental resident device-scale hybrid: device global norm/clip buffer consumed directly by AdamW/EMA; pruned component/utility tapes; CPU dispatch/four-scalar diagnostic admission/post-update summaries/full snapshots; inter-tape host transfers remain"
	}
	if scope == "ptx-resident-globalnorm-hybrid-v1" || scope == "vulkan-resident-globalnorm-hybrid-v1" {
		return "experimental resident global-norm hybrid: device gradient summary merge/clipping scale, pruned component/utility tapes and AdamW/EMA; CPU dispatch/four-scalar validation/post-update summaries/full snapshots; inter-tape host transfers remain"
	}
	if scope == "ptx-resident-pruned-hybrid-v1" || scope == "vulkan-resident-pruned-hybrid-v1" {
		return "experimental resident pruned hybrid: dependency-only execution/allocations and forward-only boundary downloads; step/branch/loss/time/sequence/projection/transformer/weighting/AdamW/EMA tapes; CPU dispatch/global norm summary reduction; transfer-inclusive boundaries/full host snapshots"
	}
	if scope == "ptx-resident-step-hybrid-v1" || scope == "vulkan-resident-step-hybrid-v1" {
		return "experimental resident step hybrid: step gradient/metric utility reduction plus branch/loss/time/sequence/projection/transformer/weighting/AdamW/EMA tapes; CPU dispatch/global norm summary reduction; transfer-inclusive boundaries/full host snapshots"
	}
	if scope == "ptx-resident-branch-hybrid-v1" || scope == "vulkan-resident-branch-hybrid-v1" {
		return "experimental resident branch hybrid: flow row interpolation/adjoint branch utility tapes plus loss/time/sequence/projection/transformer/weighting/AdamW/EMA; CPU step orchestration/branch accumulation/global norm summary reduction; transfer-inclusive boundaries/full host snapshots"
	}
	if scope == "ptx-resident-loss-hybrid-v1" || scope == "vulkan-resident-loss-hybrid-v1" {
		return "experimental resident loss hybrid: LSD/EOS losses/seeds, time/sequence/projection/flow/transformer/weighting and AdamW/EMA device tapes; CPU flow interpolation/branch orchestration/global norm summary reduction; transfer-inclusive boundaries/full host snapshots"
	}
	if scope == "ptx-resident-eos-hybrid-v1" || scope == "vulkan-resident-eos-hybrid-v1" {
		return "experimental resident EOS hybrid: stable BCE loss/weighted EOS seeds, time/sequence/projection/flow/transformer/weighting and AdamW/EMA device tapes; CPU LSD scalar losses/flow orchestration/global norm summary reduction; transfer-inclusive boundaries/full host snapshots"
	}
	if scope == "ptx-resident-time-hybrid-v1" || scope == "vulkan-resident-time-hybrid-v1" {
		return "experimental resident time hybrid: sinusoidal time values/JVP/adjoints, embedding/BOS/sequence, projections, flow/transformer/weighting and AdamW/EMA device tapes; CPU scalar losses and global norm summary reduction; transfer-inclusive boundaries and full host snapshot per completed update"
	}
	if scope == "ptx-resident-sequence-hybrid-v1" || scope == "vulkan-resident-sequence-hybrid-v1" {
		return "experimental resident sequence hybrid: embedding/BOS/sequence, speaker/audio/EOS, flow/transformer/weighting tapes and AdamW/EMA on device; CPU scalar losses/time embeddings and global norm summary reduction; transfer-inclusive boundaries and full host snapshot per completed update"
	}
	if scope == "ptx-resident-projection-hybrid-v1" || scope == "vulkan-resident-projection-hybrid-v1" {
		return "experimental resident projection hybrid: speaker/audio/EOS, flow/transformer/weighting tapes and AdamW/EMA on device; CPU embedding/sequence assembly, scalar losses/time embeddings and global norm summary reduction; full host snapshot per completed update"
	}
	if scope == "ptx-resident-weight-hybrid-v1" || scope == "vulkan-resident-weight-hybrid-v1" {
		return "experimental resident weighting hybrid: flow/transformer/weighting tapes and AdamW/EMA on device; CPU sequence/conditioner/EOS, scalar losses/time embeddings and global norm summary reduction; full host snapshot per completed update"
	}
	if scope == "ptx-resident-hybrid-v1" || scope == "vulkan-resident-hybrid-v1" {
		return "experimental resident hybrid: flow and transformer tapes bind resident parameters and accumulate gradients on device; resident AdamW/EMA; CPU sequence/conditioner/EOS/weighting, scalar losses/time embeddings and global norm summary reduction; full host snapshot per completed update"
	}
	if scope == "ptx-optimizer-hybrid-v1" || scope == "vulkan-optimizer-hybrid-v1" {
		return "experimental optimizer hybrid: CPU forward/loss/backward; resident device accumulation, chunk norm, AdamW/EMA; host global norm reduction and full state synchronisation per update; checkpoints use coherent host snapshots"
	}
	if scope == "ptx-transformer-hybrid-v1" || scope == "vulkan-transformer-hybrid-v1" {
		return "experimental transformer hybrid: transformer forward and parameter/input VJP use device tapes; sequence assembly, conditioner/EOS/flow, loss reductions, AdamW/EMA and checkpoints run on CPU; parameters transfer twice per update"
	}
	if scope == "ptx-flow-hybrid-v1" || scope == "vulkan-flow-hybrid-v1" {
		return "experimental flow hybrid: flow forward/time-JVP and adjoints use device tapes; time embeddings, transformer, scalar losses, AdamW/EMA and checkpointing run on CPU"
	}
	return "experimental attention-backward hybrid: forward, other gradients, AdamW/EMA and checkpointing run on CPU"
}

// Native initialisation occurs only after explicit admission and local input
// validation. Merely importing this file does not load a GPU driver/runtime.
func openTrainingAttentionBackend(ctx context.Context, scope string, maxBytes int64) (pockettts.TrainingAttentionBackend, func() error, error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("nil training context")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	switch scope {
	case "simd":
		return nil, func() error { return nil }, nil
	case "ptx-attention-hybrid-v1":
		op, err := nvidia.NewTrainingF32()
		if err != nil {
			if op != nil {
				return nil, op.Close, err
			}
			return nil, nil, err
		}
		backend := &pockettts.PTXTrainingAttention{Context: ctx, Operator: op, MaxBytes: maxBytes}
		return backend, backend.Close, nil
	case "vulkan-attention-hybrid-v1":
		if !vulkan.VulkanInit() {
			return nil, nil, fmt.Errorf("Vulkan unavailable")
		}
		op, err := vulkan.NewVkTrainingF32(ctx)
		if err != nil {
			if op != nil {
				return nil, op.Close, err
			}
			return nil, nil, err
		}
		backend := &pockettts.VulkanTrainingAttention{Context: ctx, Operator: op, MaxBytes: maxBytes}
		return backend, backend.Close, nil
	default:
		return nil, nil, fmt.Errorf("unsupported training backend scope")
	}
}
