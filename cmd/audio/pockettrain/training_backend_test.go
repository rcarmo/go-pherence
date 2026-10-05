//go:build linux

package main

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"strconv"
	"strings"
	"testing"
)

func TestPreparedHybridBackendAdmission(t *testing.T) {
	for _, name := range []string{"ptx-attention-hybrid", "vulkan-attention-hybrid", "ptx-flow-hybrid", "vulkan-flow-hybrid", "ptx-transformer-hybrid", "vulkan-transformer-hybrid", "ptx-optimizer-hybrid", "vulkan-optimizer-hybrid", "ptx-resident-hybrid", "vulkan-resident-hybrid", "ptx-resident-weight-hybrid", "vulkan-resident-weight-hybrid", "ptx-resident-projection-hybrid", "vulkan-resident-projection-hybrid", "ptx-resident-sequence-hybrid", "vulkan-resident-sequence-hybrid", "ptx-resident-time-hybrid", "vulkan-resident-time-hybrid", "ptx-resident-eos-hybrid", "vulkan-resident-eos-hybrid", "ptx-resident-loss-hybrid", "vulkan-resident-loss-hybrid", "ptx-resident-branch-hybrid", "vulkan-resident-branch-hybrid", "ptx-resident-step-hybrid", "vulkan-resident-step-hybrid", "ptx-resident-pruned-hybrid", "vulkan-resident-pruned-hybrid", "ptx-resident-globalnorm-hybrid", "ptx-resident-devicescale-hybrid", "ptx-resident-sampleweight-hybrid", "ptx-resident-lifetime-hybrid", "ptx-resident-statecheck-hybrid", "ptx-resident-zeroconst-hybrid", "ptx-resident-eosfused-hybrid", "ptx-resident-audiofused-hybrid", "ptx-resident-voicefused-hybrid", "ptx-resident-conditioner-hybrid", "ptx-resident-backbone-hybrid", "ptx-resident-eosbackbone-hybrid", "ptx-resident-diagfused-hybrid", "ptx-resident-endpointfused-hybrid", "ptx-resident-primaryfused-hybrid", "ptx-resident-distillfused-hybrid", "ptx-resident-forwardfused-hybrid", "ptx-resident-diagforward-hybrid", "ptx-resident-weightroots-hybrid", "ptx-resident-stepweight-hybrid", "ptx-resident-stepbackbone-hybrid", "ptx-resident-stepresults-hybrid", "ptx-resident-distillresults-hybrid", "ptx-resident-eosforward-hybrid", "ptx-resident-forwardresults-hybrid", "ptx-resident-embeddingindex-hybrid", "ptx-resident-embeddingcompact-hybrid", "ptx-resident-diagadmission-hybrid", "ptx-resident-distilladmission-hybrid", "vulkan-resident-globalnorm-hybrid", "vulkan-resident-devicescale-hybrid", "vulkan-resident-backbone-hybrid", "vulkan-resident-eosbackbone-hybrid", "vulkan-resident-diagfused-hybrid", "vulkan-resident-endpointfused-hybrid", "vulkan-resident-primaryfused-hybrid", "vulkan-resident-distillfused-hybrid", "vulkan-resident-forwardfused-hybrid", "vulkan-resident-diagforward-hybrid", "vulkan-resident-weightroots-hybrid", "vulkan-resident-stepweight-hybrid", "vulkan-resident-stepbackbone-hybrid", "vulkan-resident-stepresults-hybrid", "vulkan-resident-distillresults-hybrid", "vulkan-resident-eosforward-hybrid", "vulkan-resident-forwardresults-hybrid", "vulkan-resident-embeddingindex-hybrid", "vulkan-resident-embeddingcompact-hybrid", "vulkan-resident-diagadmission-hybrid", "vulkan-resident-distilladmission-hybrid"} {
		if _, err := trainingBackendScope(name, false, strings.Repeat("a", 64), 1<<20); err == nil {
			t.Fatal("no native optin")
		}
		if _, err := trainingBackendScope(name, true, "bad", 1<<20); err == nil {
			t.Fatal("no runtime pin")
		}
		for _, ceiling := range []int64{0, -1, 2<<30 + 1} {
			if _, err := trainingBackendScope(name, true, strings.Repeat("a", 64), ceiling); err == nil {
				t.Fatal("invalid byte ceiling", ceiling)
			}
		}
		if err := trainPrepared(context.Background(), []string{"-backend=" + name, "-allow-experimental-device-training", "-device-runtime-sha256=" + strings.Repeat("a", 64), "-config=/missing"}, io.Discard); err == nil || !strings.Contains(err.Error(), "allow-experimental-sampler") {
			t.Fatal("sampler admission before file/native", err)
		}
		scope, err := trainingBackendScope(name, true, strings.Repeat("a", 64), 1<<20)
		if err != nil || scope != name+"-v1" {
			t.Fatal(scope, err)
		}
		if err := trainPrepared(context.Background(), []string{"-backend=" + name, "-config=/missing"}, io.Discard); err == nil || !strings.Contains(err.Error(), "opt-in") {
			t.Fatal("hybrid admission before file/native", err)
		}
	}
	owners, cleanup, err := openTrainingBackend(context.Background(), "simd", 0)
	if err != nil || owners.attention != nil || owners.flow != nil || owners.transformer != nil || owners.optimizer != nil || owners.weight != nil || owners.projection != nil || owners.sequence != nil || owners.eos != nil || owners.loss != nil || owners.branch != nil || owners.step != nil || cleanup == nil {
		t.Fatal(err)
	}
	if err := cleanup(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := openTrainingBackend(context.Background(), "unknown", 1<<20); err == nil {
		t.Fatal("unknown backend")
	}
}

func TestPreparedFlowBackendFactoryOffline(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, scope := range []string{"simd", "ptx-flow-hybrid-v1", "vulkan-flow-hybrid-v1", "ptx-attention-hybrid-v1", "vulkan-attention-hybrid-v1", "ptx-transformer-hybrid-v1", "vulkan-transformer-hybrid-v1", "ptx-optimizer-hybrid-v1", "vulkan-optimizer-hybrid-v1", "ptx-resident-hybrid-v1", "vulkan-resident-hybrid-v1", "ptx-resident-weight-hybrid-v1", "vulkan-resident-weight-hybrid-v1", "ptx-resident-projection-hybrid-v1", "vulkan-resident-projection-hybrid-v1", "ptx-resident-sequence-hybrid-v1", "vulkan-resident-sequence-hybrid-v1", "ptx-resident-time-hybrid-v1", "vulkan-resident-time-hybrid-v1", "ptx-resident-eos-hybrid-v1", "vulkan-resident-eos-hybrid-v1", "ptx-resident-loss-hybrid-v1", "vulkan-resident-loss-hybrid-v1", "ptx-resident-branch-hybrid-v1", "vulkan-resident-branch-hybrid-v1", "ptx-resident-step-hybrid-v1", "vulkan-resident-step-hybrid-v1", "ptx-resident-pruned-hybrid-v1", "vulkan-resident-pruned-hybrid-v1", "ptx-resident-globalnorm-hybrid-v1", "ptx-resident-devicescale-hybrid-v1", "ptx-resident-sampleweight-hybrid-v1", "ptx-resident-lifetime-hybrid-v1", "ptx-resident-statecheck-hybrid-v1", "ptx-resident-zeroconst-hybrid-v1", "ptx-resident-eosfused-hybrid-v1", "ptx-resident-audiofused-hybrid-v1", "ptx-resident-voicefused-hybrid-v1", "ptx-resident-conditioner-hybrid-v1", "ptx-resident-backbone-hybrid-v1", "ptx-resident-eosbackbone-hybrid-v1", "ptx-resident-diagfused-hybrid-v1", "ptx-resident-endpointfused-hybrid-v1", "ptx-resident-primaryfused-hybrid-v1", "ptx-resident-distillfused-hybrid-v1", "ptx-resident-forwardfused-hybrid-v1", "ptx-resident-diagforward-hybrid-v1", "ptx-resident-weightroots-hybrid-v1", "ptx-resident-stepweight-hybrid-v1", "ptx-resident-stepbackbone-hybrid-v1", "ptx-resident-stepresults-hybrid-v1", "ptx-resident-distillresults-hybrid-v1", "ptx-resident-eosforward-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1", "ptx-resident-embeddingindex-hybrid-v1", "ptx-resident-embeddingcompact-hybrid-v1", "ptx-resident-diagadmission-hybrid-v1", "ptx-resident-distilladmission-hybrid-v1", "vulkan-resident-embeddingindex-hybrid-v1", "vulkan-resident-embeddingcompact-hybrid-v1", "vulkan-resident-diagadmission-hybrid-v1", "vulkan-resident-distilladmission-hybrid-v1", "vulkan-resident-globalnorm-hybrid-v1", "vulkan-resident-devicescale-hybrid-v1"} {
		if _, cleanup, err := openTrainingBackend(nil, scope, 1<<20); err == nil || cleanup != nil {
			t.Fatal("nil context admitted", scope)
		}
		if _, cleanup, err := openTrainingBackend(ctx, scope, 1<<20); err != context.Canceled || cleanup != nil {
			t.Fatal("cancelled factory entered", scope, err)
		}
		if scope != "simd" {
			if _, cleanup, err := openTrainingBackend(context.Background(), scope, 0); err == nil || cleanup != nil {
				t.Fatal("invalid ceiling admitted", scope)
			}
		}
	}
	for _, name := range []string{"ptx", "vulkan", "unknown", "ptx-flow-hybrid-v2"} {
		if _, err := trainingBackendScope(name, true, strings.Repeat("a", 64), 1<<20); err == nil {
			t.Fatal("unsupported admitted", name)
		}
	}
}
func TestPreparedFlowBackendReplayOptions(t *testing.T) {
	if scope, err := trainingBackendScope("simd", false, "", 0); err != nil || scope != "simd" {
		t.Fatal("CPU defaults changed")
	}
	sentinel := errors.New("native cleanup failed")
	owner := &struct{}{}
	retained := &trainingResourcesRetainedError{err: sentinel, owner: owner}
	if !errors.Is(retained, sentinel) || !strings.Contains(retained.Error(), "resources retained") || retained.owner != owner {
		t.Fatal("error lost owner/cause")
	}
	base, pin := strings.Repeat("7", 64), strings.Repeat("a", 64)
	if got := trainingBackendOptionsHash(base, "simd", pin); got != base {
		t.Fatal("SIMD identity changed")
	}
	seen := map[string]bool{}
	for _, scope := range []string{"ptx-attention-hybrid-v1", "vulkan-attention-hybrid-v1", "ptx-flow-hybrid-v1", "vulkan-flow-hybrid-v1", "ptx-transformer-hybrid-v1", "vulkan-transformer-hybrid-v1", "ptx-optimizer-hybrid-v1", "vulkan-optimizer-hybrid-v1", "ptx-resident-hybrid-v1", "vulkan-resident-hybrid-v1", "ptx-resident-weight-hybrid-v1", "vulkan-resident-weight-hybrid-v1", "ptx-resident-projection-hybrid-v1", "vulkan-resident-projection-hybrid-v1", "ptx-resident-sequence-hybrid-v1", "vulkan-resident-sequence-hybrid-v1", "ptx-resident-time-hybrid-v1", "vulkan-resident-time-hybrid-v1", "ptx-resident-eos-hybrid-v1", "vulkan-resident-eos-hybrid-v1", "ptx-resident-loss-hybrid-v1", "vulkan-resident-loss-hybrid-v1", "ptx-resident-branch-hybrid-v1", "vulkan-resident-branch-hybrid-v1", "ptx-resident-step-hybrid-v1", "vulkan-resident-step-hybrid-v1", "ptx-resident-pruned-hybrid-v1", "vulkan-resident-pruned-hybrid-v1", "ptx-resident-globalnorm-hybrid-v1", "ptx-resident-devicescale-hybrid-v1", "ptx-resident-sampleweight-hybrid-v1", "ptx-resident-lifetime-hybrid-v1", "ptx-resident-statecheck-hybrid-v1", "ptx-resident-zeroconst-hybrid-v1", "ptx-resident-eosfused-hybrid-v1", "ptx-resident-audiofused-hybrid-v1", "ptx-resident-voicefused-hybrid-v1", "ptx-resident-conditioner-hybrid-v1", "ptx-resident-backbone-hybrid-v1", "ptx-resident-eosbackbone-hybrid-v1", "ptx-resident-diagfused-hybrid-v1", "ptx-resident-endpointfused-hybrid-v1", "ptx-resident-primaryfused-hybrid-v1", "ptx-resident-distillfused-hybrid-v1", "ptx-resident-forwardfused-hybrid-v1", "ptx-resident-diagforward-hybrid-v1", "ptx-resident-weightroots-hybrid-v1", "ptx-resident-stepweight-hybrid-v1", "ptx-resident-stepbackbone-hybrid-v1", "ptx-resident-stepresults-hybrid-v1", "ptx-resident-distillresults-hybrid-v1", "ptx-resident-eosforward-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1", "ptx-resident-embeddingindex-hybrid-v1", "ptx-resident-embeddingcompact-hybrid-v1", "ptx-resident-diagadmission-hybrid-v1", "ptx-resident-distilladmission-hybrid-v1", "vulkan-resident-globalnorm-hybrid-v1", "vulkan-resident-devicescale-hybrid-v1"} {
		got := trainingBackendOptionsHash(base, scope, pin)
		if seen[got] || got == base {
			t.Fatal("scope not bound")
		}
		seen[got] = true
		if trainingBackendOptionsHash(base, scope, strings.Repeat("b", 64)) == got {
			t.Fatal("runtime pin not bound")
		}
		if strings.Contains(scope, "attention") {
			if got != hashText(base+";attention-backend="+scope+";device-runtime="+pin) {
				t.Fatal("attention v1 compatibility changed")
			}
		}
		notice := trainingBackendNotice(scope)
		if !strings.Contains(notice, "CPU") || !strings.Contains(notice, "AdamW/EMA") {
			t.Fatal("notice scope")
		}
		if strings.Contains(scope, "flow") && !strings.Contains(notice, "transformer") {
			t.Fatal("flow scope unclear")
		}
	}
}

// Inspect factory cases rather than initializing native owners. This regression
// detects admitted scopes omitted from the switch, even with GPUs held disabled.
func TestPreparedResidentFactoryScopeCoverage(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "training_backend.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok {
			return true
		}
		if fn.Name.Name != "openTrainingBackend" {
			return false
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			c, ok := n.(*ast.CaseClause)
			if !ok {
				return true
			}
			for _, e := range c.List {
				if s, ok := e.(*ast.BasicLit); ok && s.Kind == token.STRING {
					v, err := strconv.Unquote(s.Value)
					if err != nil {
						t.Fatal(err)
					}
					found[v] = true
				}
			}
			return true
		})
		return false
	})
	for _, kind := range []string{"ptx", "vulkan"} {
		for _, mode := range []string{"sampleweight", "lifetime", "statecheck", "zeroconst", "eosfused", "audiofused", "voicefused", "conditioner", "backbone", "eosbackbone", "diagfused", "endpointfused", "primaryfused", "distillfused", "forwardfused", "diagforward", "weightroots", "stepweight", "stepbackbone", "stepresults", "distillresults", "eosforward", "forwardresults", "embeddingindex", "embeddingcompact", "diagadmission", "distilladmission", "flowhandoff", "eosadmission", "predictionhandoff", "devicehandoff"} {
			scope := kind + "-resident-" + mode + "-hybrid-v1"
			if !found[scope] {
				t.Fatal("missing factory case", scope)
			}
		}
	}
	base, pin := strings.Repeat("a", 64), strings.Repeat("b", 64)
	for _, kind := range []string{"ptx", "vulkan"} {
		scope := kind + "-resident-backbone-hybrid-v1"
		if trainingBackendOptionsHash(base, scope, pin) == trainingBackendOptionsHash(base, kind+"-resident-conditioner-hybrid-v1", pin) {
			t.Fatal("replay collision")
		}
		if trainingBackendOptionsHash(base, scope, pin) != hashText(base+";resident-backbone-backend="+scope+";device-runtime="+pin) {
			t.Fatal("backbone hash identity")
		}
		eosScope := kind + "-resident-eosbackbone-hybrid-v1"
		if trainingBackendOptionsHash(base, eosScope, pin) == trainingBackendOptionsHash(base, scope, pin) || trainingBackendOptionsHash(base, eosScope, pin) != hashText(base+";resident-eosbackbone-backend="+eosScope+";device-runtime="+pin) {
			t.Fatal("EOS backbone replay identity")
		}
		diagScope := kind + "-resident-diagfused-hybrid-v1"
		if trainingBackendOptionsHash(base, diagScope, pin) == trainingBackendOptionsHash(base, eosScope, pin) || trainingBackendOptionsHash(base, diagScope, pin) != hashText(base+";resident-diagfused-backend="+diagScope+";device-runtime="+pin) {
			t.Fatal("diagonal replay identity")
		}
		endpointScope := kind + "-resident-endpointfused-hybrid-v1"
		if trainingBackendOptionsHash(base, endpointScope, pin) == trainingBackendOptionsHash(base, diagScope, pin) || trainingBackendOptionsHash(base, endpointScope, pin) != hashText(base+";resident-endpointfused-backend="+endpointScope+";device-runtime="+pin) {
			t.Fatal("endpoint replay identity")
		}
		primaryScope := kind + "-resident-primaryfused-hybrid-v1"
		if trainingBackendOptionsHash(base, primaryScope, pin) == trainingBackendOptionsHash(base, endpointScope, pin) || trainingBackendOptionsHash(base, primaryScope, pin) != hashText(base+";resident-primaryfused-backend="+primaryScope+";device-runtime="+pin) {
			t.Fatal("primary replay identity")
		}
		distillScope := kind + "-resident-distillfused-hybrid-v1"
		if trainingBackendOptionsHash(base, distillScope, pin) == trainingBackendOptionsHash(base, primaryScope, pin) || trainingBackendOptionsHash(base, distillScope, pin) != hashText(base+";resident-distillfused-backend="+distillScope+";device-runtime="+pin) {
			t.Fatal("distill replay identity")
		}
		forwardScope := kind + "-resident-forwardfused-hybrid-v1"
		if trainingBackendOptionsHash(base, forwardScope, pin) == trainingBackendOptionsHash(base, distillScope, pin) || trainingBackendOptionsHash(base, forwardScope, pin) != hashText(base+";resident-forwardfused-backend="+forwardScope+";device-runtime="+pin) {
			t.Fatal("forward replay identity")
		}
		diagForwardScope := kind + "-resident-diagforward-hybrid-v1"
		if trainingBackendOptionsHash(base, diagForwardScope, pin) == trainingBackendOptionsHash(base, forwardScope, pin) || trainingBackendOptionsHash(base, diagForwardScope, pin) != hashText(base+";resident-diagforward-backend="+diagForwardScope+";device-runtime="+pin) {
			t.Fatal("diagforward replay identity")
		}
		weightRootsScope := kind + "-resident-weightroots-hybrid-v1"
		if trainingBackendOptionsHash(base, weightRootsScope, pin) == trainingBackendOptionsHash(base, diagForwardScope, pin) || trainingBackendOptionsHash(base, weightRootsScope, pin) != hashText(base+";resident-weightroots-backend="+weightRootsScope+";device-runtime="+pin) {
			t.Fatal("weightroots replay identity")
		}
		stepWeightScope := kind + "-resident-stepweight-hybrid-v1"
		if trainingBackendOptionsHash(base, stepWeightScope, pin) == trainingBackendOptionsHash(base, weightRootsScope, pin) || trainingBackendOptionsHash(base, stepWeightScope, pin) != hashText(base+";resident-stepweight-backend="+stepWeightScope+";device-runtime="+pin) {
			t.Fatal("stepweight replay identity")
		}
		stepBackboneScope := kind + "-resident-stepbackbone-hybrid-v1"
		if trainingBackendOptionsHash(base, stepBackboneScope, pin) == trainingBackendOptionsHash(base, stepWeightScope, pin) || trainingBackendOptionsHash(base, stepBackboneScope, pin) != hashText(base+";resident-stepbackbone-backend="+stepBackboneScope+";device-runtime="+pin) {
			t.Fatal("stepbackbone replay identity")
		}
		stepResultsScope := kind + "-resident-stepresults-hybrid-v1"
		if trainingBackendOptionsHash(base, stepResultsScope, pin) == trainingBackendOptionsHash(base, stepBackboneScope, pin) || trainingBackendOptionsHash(base, stepResultsScope, pin) != hashText(base+";resident-stepresults-backend="+stepResultsScope+";device-runtime="+pin) {
			t.Fatal("stepresults replay identity")
		}
		distillResultsScope := kind + "-resident-distillresults-hybrid-v1"
		if trainingBackendOptionsHash(base, distillResultsScope, pin) == trainingBackendOptionsHash(base, stepResultsScope, pin) || trainingBackendOptionsHash(base, distillResultsScope, pin) != hashText(base+";resident-distillresults-backend="+distillResultsScope+";device-runtime="+pin) {
			t.Fatal("distillresults replay identity")
		}
		eosForwardScope := kind + "-resident-eosforward-hybrid-v1"
		if trainingBackendOptionsHash(base, eosForwardScope, pin) == trainingBackendOptionsHash(base, distillResultsScope, pin) || trainingBackendOptionsHash(base, eosForwardScope, pin) != hashText(base+";resident-eosforward-backend="+eosForwardScope+";device-runtime="+pin) {
			t.Fatal("eosforward replay identity")
		}
		forwardResultsScope := kind + "-resident-forwardresults-hybrid-v1"
		if trainingBackendOptionsHash(base, forwardResultsScope, pin) == trainingBackendOptionsHash(base, eosForwardScope, pin) || trainingBackendOptionsHash(base, forwardResultsScope, pin) != hashText(base+";resident-forwardresults-backend="+forwardResultsScope+";device-runtime="+pin) {
			t.Fatal("forwardresults replay identity")
		}
	}
}

func TestPreparedResidentEmbeddingIndexIdentity(t *testing.T) {
	base := hashText("embedding-index-options")
	pin := strings.Repeat("b", 64)
	for _, kind := range []string{"ptx", "vulkan"} {
		scope := kind + "-resident-embeddingindex-hybrid-v1"
		if got := trainingBackendOptionsHash(base, scope, pin); got != hashText(base+";resident-embeddingindex-backend="+scope+";device-runtime="+pin) {
			t.Fatal("indexed identity", got)
		}
		if strings.Contains(trainingBackendNotice(scope), "dense gradients") == false {
			t.Fatal("scope overclaim")
		}
		if _, err := trainingBackendScope(kind+"-resident-embeddingindex-hybrid", false, pin, 1<<20); err == nil {
			t.Fatal("opt-in")
		}
	}
}

func TestPreparedResidentEmbeddingCompactIdentity(t *testing.T) {
	base := hashText("compact-embedding-options")
	pin := strings.Repeat("b", 64)
	for _, kind := range []string{"ptx", "vulkan"} {
		scope := kind + "-resident-embeddingcompact-hybrid-v1"
		if got := trainingBackendOptionsHash(base, scope, pin); got != hashText(base+";resident-embeddingcompact-backend="+scope+";device-runtime="+pin) {
			t.Fatal("compact identity", got)
		}
		if !strings.Contains(trainingBackendNotice(scope), "dense resident gradient") {
			t.Fatal("scope overclaim")
		}
		if _, err := trainingBackendScope(kind+"-resident-embeddingcompact-hybrid", false, pin, 1<<20); err == nil {
			t.Fatal("optin")
		}
	}
}

func TestPreparedResidentDiagAdmissionIdentity(t *testing.T) {
	base := hashText("diagonal-admission-options")
	pin := strings.Repeat("b", 64)
	for _, kind := range []string{"ptx", "vulkan"} {
		scope := kind + "-resident-diagadmission-hybrid-v1"
		if got := trainingBackendOptionsHash(base, scope, pin); got != hashText(base+";resident-diagadmission-backend="+scope+";device-runtime="+pin) {
			t.Fatal("guard identity")
		}
		if !strings.Contains(trainingBackendNotice(scope), "prediction downloads") {
			t.Fatal("overclaim")
		}
		if _, err := trainingBackendScope(kind+"-resident-diagadmission-hybrid", false, pin, 1<<20); err == nil {
			t.Fatal("opt-in")
		}
	}
}

func TestPreparedResidentDistillAdmissionIdentity(t *testing.T) {
	base := hashText("distillation-admission-options")
	pin := strings.Repeat("b", 64)
	for _, kind := range []string{"ptx", "vulkan"} {
		scope := kind + "-resident-distilladmission-hybrid-v1"
		if got := trainingBackendOptionsHash(base, scope, pin); got != hashText(base+";resident-distilladmission-backend="+scope+";device-runtime="+pin) {
			t.Fatal("guard identity")
		}
		if !strings.Contains(trainingBackendNotice(scope), "prediction downloads") {
			t.Fatal("overclaim")
		}
		if _, err := trainingBackendScope(kind+"-resident-distilladmission-hybrid", false, pin, 1<<20); err == nil {
			t.Fatal("opt-in")
		}
	}
}
