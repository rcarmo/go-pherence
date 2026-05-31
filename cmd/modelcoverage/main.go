package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
)

type manifest struct {
	Version  int                       `json:"version"`
	Families map[string]manifestFamily `json:"families"`
}

type manifestFamily struct {
	Status            string          `json:"status"`
	RuntimeGeneration bool            `json:"runtime_generation"`
	ValidationTarget  string          `json:"validation_target"`
	Packages          []string        `json:"packages"`
	Coverage          map[string]bool `json:"coverage"`
	Commands          []string        `json:"commands"`
}

type familySummary struct {
	Name              string                    `json:"name"`
	Status            string                    `json:"status"`
	RuntimeGeneration bool                      `json:"runtime_generation"`
	ValidationTarget  string                    `json:"validation_target"`
	Covered           int                       `json:"covered"`
	Pending           int                       `json:"pending"`
	CoveragePercent   float64                   `json:"coverage_percent"`
	PendingKeys       []string                  `json:"pending_keys,omitempty"`
	Categories        map[string]categoryCounts `json:"categories,omitempty"`
}

type categoryCounts struct {
	Covered     int      `json:"covered"`
	Pending     int      `json:"pending"`
	Percent     float64  `json:"percent"`
	PendingKeys []string `json:"pending_keys,omitempty"`
}

type runtimeRoadmapFamily struct {
	Family   string                  `json:"family"`
	Blockers []runtimeRoadmapBlocker `json:"blockers"`
}

type runtimeRoadmapBlocker struct {
	Key           string `json:"key"`
	Phase         int    `json:"phase"`
	Description   string `json:"description"`
	Package       string `json:"package,omitempty"`
	Prerequisites string `json:"prerequisites,omitempty"`
	Validation    string `json:"validation,omitempty"`
}

func main() {
	manifestPath := flag.String("manifest", "docs/model-coverage-manifest.json", "model coverage manifest path")
	family := flag.String("family", "", "optional family name to summarize, e.g. qwen3_tts or lfm2_moe")
	jsonOut := flag.Bool("json", false, "emit JSON summary")
	markdownOut := flag.Bool("markdown", false, "emit Markdown summary table")
	csvOut := flag.Bool("csv", false, "emit CSV summary rows")
	runtimeRoadmap := flag.Bool("runtime-roadmap", false, "emit Markdown checklist of pending runtime/backend gates")
	runtimeRoadmapJSON := flag.Bool("runtime-roadmap-json", false, "emit JSON runtime blocker roadmap")
	nextRuntime := flag.Bool("next-runtime", false, "emit the next dependency-ordered runtime blocker per family")
	nextRuntimeJSON := flag.Bool("next-runtime-json", false, "emit JSON for the next dependency-ordered runtime blocker per family")
	blockerPackage := flag.String("blocker-package", "", "optional package path filter for runtime roadmap/next-runtime outputs")
	snapshotOut := flag.Bool("snapshot", false, "emit Markdown coverage snapshot plus runtime roadmap")
	pendingOnly := flag.Bool("pending-only", false, "only print pending coverage gate names in text mode")
	failPending := flag.Bool("fail-pending", false, "exit non-zero if any selected coverage gates are pending")
	minPercent := flag.Float64("min-percent", -1, "exit non-zero if any selected family coverage percent is below this threshold")
	referencesOnly := flag.Bool("references-only", false, "only include reference/fixture coverage gates in counts and pending output")
	runtimeOnly := flag.Bool("runtime-only", false, "only include runtime/backend coverage gates in counts and pending output")
	executionOnly := flag.Bool("execution-only", false, "only include actual execution/runtime implementation gates in counts and pending output")
	parityOnly := flag.Bool("parity-only", false, "only include numeric parity/reference readiness gates in counts and pending output")
	readinessOnly := flag.Bool("readiness-only", false, "only include readiness/report/gate coverage keys in counts and pending output")
	flag.Parse()
	m, err := loadManifest(*manifestPath)
	if err != nil {
		fatal(err)
	}
	summaries, err := summarize(m, *family, coverageFilter{ReferencesOnly: *referencesOnly, RuntimeOnly: *runtimeOnly, ExecutionOnly: *executionOnly, ParityOnly: *parityOnly, ReadinessOnly: *readinessOnly})
	if err != nil {
		fatal(err)
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summaries); err != nil {
			fatal(err)
		}
	} else if *markdownOut {
		printMarkdownSummary(os.Stdout, summaries)
	} else if *csvOut {
		printCSVSummary(os.Stdout, summaries)
	} else if *runtimeRoadmap {
		printRuntimeRoadmap(os.Stdout, summaries, *blockerPackage)
	} else if *runtimeRoadmapJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(buildRuntimeRoadmap(summaries, *blockerPackage)); err != nil {
			fatal(err)
		}
	} else if *nextRuntime {
		printNextRuntime(os.Stdout, summaries, *blockerPackage)
	} else if *nextRuntimeJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(buildNextRuntime(summaries, *blockerPackage)); err != nil {
			fatal(err)
		}
	} else if *snapshotOut {
		printSnapshot(os.Stdout, summaries)
	} else {
		for _, s := range summaries {
			if *pendingOnly {
				for _, key := range s.PendingKeys {
					fmt.Printf("%s.%s\n", s.Name, key)
				}
				continue
			}
			printTextSummary(os.Stdout, s)
		}
	}
	if *failPending {
		for _, s := range summaries {
			if s.Pending > 0 {
				os.Exit(1)
			}
		}
	}
	if *minPercent >= 0 && !summariesMeetMinPercent(summaries, *minPercent) {
		os.Exit(1)
	}
}

func buildNextRuntime(summaries []familySummary, packageFilter string) []runtimeRoadmapFamily {
	var next []runtimeRoadmapFamily
	for _, family := range buildRuntimeRoadmap(summaries, packageFilter) {
		if len(family.Blockers) == 0 {
			continue
		}
		next = append(next, runtimeRoadmapFamily{Family: family.Family, Blockers: []runtimeRoadmapBlocker{family.Blockers[0]}})
	}
	return next
}

func printNextRuntime(w interface{ Write([]byte) (int, error) }, summaries []familySummary, packageFilter string) {
	for _, family := range buildNextRuntime(summaries, packageFilter) {
		blocker := family.Blockers[0]
		fmt.Fprintf(w, "%s.%s — %s", family.Family, blocker.Key, blocker.Description)
		if blocker.Prerequisites != "" {
			fmt.Fprintf(w, " (after: %s)", blocker.Prerequisites)
		}
		if blocker.Package != "" {
			fmt.Fprintf(w, " (package: %s)", blocker.Package)
		}
		if blocker.Validation != "" {
			fmt.Fprintf(w, " (validate: %s)", blocker.Validation)
		}
		fmt.Fprintln(w)
	}
}

func buildRuntimeRoadmap(summaries []familySummary, packageFilter string) []runtimeRoadmapFamily {
	roadmap := make([]runtimeRoadmapFamily, 0, len(summaries))
	for _, s := range summaries {
		pending := orderedRuntimePending(s.Categories["runtime"].PendingKeys)
		if len(pending) == 0 {
			continue
		}
		family := runtimeRoadmapFamily{Family: s.Name, Blockers: make([]runtimeRoadmapBlocker, 0, len(pending))}
		for _, key := range pending {
			blocker := runtimeRoadmapBlocker{Key: key, Phase: runtimeBlockerPriority(key), Description: runtimeBlockerDescription(key), Package: runtimeBlockerPackage(key), Prerequisites: runtimeBlockerPrerequisites(key), Validation: runtimeBlockerValidation(key)}
			if packageFilter != "" && blocker.Package != packageFilter {
				continue
			}
			family.Blockers = append(family.Blockers, blocker)
		}
		if len(family.Blockers) == 0 {
			continue
		}
		roadmap = append(roadmap, family)
	}
	return roadmap
}

func printRuntimeRoadmap(w interface{ Write([]byte) (int, error) }, summaries []familySummary, packageFilter string) {
	for _, family := range buildRuntimeRoadmap(summaries, packageFilter) {
		fmt.Fprintf(w, "## %s runtime blockers\n\n", family.Family)
		for _, blocker := range family.Blockers {
			fmt.Fprintf(w, "- [ ] P%d `%s` — %s", blocker.Phase, blocker.Key, blocker.Description)
			if blocker.Package != "" {
				fmt.Fprintf(w, " _(package: `%s`)_", blocker.Package)
			}
			if blocker.Prerequisites != "" {
				fmt.Fprintf(w, " _(after: %s)_", blocker.Prerequisites)
			}
			if blocker.Validation != "" {
				fmt.Fprintf(w, " _(validate: `%s`)_", blocker.Validation)
			}
			fmt.Fprintln(w)
		}
		fmt.Fprintln(w)
	}
}

func orderedRuntimePending(keys []string) []string {
	out := append([]string(nil), keys...)
	sort.SliceStable(out, func(i, j int) bool {
		pi, pj := runtimeBlockerPriority(out[i]), runtimeBlockerPriority(out[j])
		if pi != pj {
			return pi < pj
		}
		return out[i] < out[j]
	})
	return out
}

func runtimeBlockerPriority(key string) int {
	switch key {
	case "cpu_talker_runtime", "cpu_generation_runtime":
		return 10
	case "cpu_code_predictor_runtime":
		return 20
	case "decoder12hz_runtime":
		return 30
	case "nvidia_runtime":
		return 90
	case "streaming_runtime":
		return 100
	default:
		return 50
	}
}

func runtimeBlockerPrerequisites(key string) string {
	switch key {
	case "cpu_code_predictor_runtime":
		return "cpu_talker_runtime"
	case "decoder12hz_runtime":
		return "cpu_code_predictor_runtime"
	case "nvidia_runtime":
		return "CPU/reference parity"
	case "streaming_runtime":
		return "CPU/reference parity, nvidia_runtime where applicable"
	default:
		return ""
	}
}

func runtimeBlockerPackage(key string) string {
	switch key {
	case "cpu_talker_runtime", "cpu_code_predictor_runtime", "decoder12hz_runtime", "streaming_runtime":
		return "model/qwen3tts"
	case "cpu_generation_runtime":
		return "model/lfm2"
	case "nvidia_runtime":
		return "backends/nvidia"
	default:
		return ""
	}
}

func runtimeBlockerValidation(key string) string {
	switch key {
	case "cpu_talker_runtime":
		return "cmd/qwen3ttsinspect -require-numeric-parity"
	case "cpu_code_predictor_runtime":
		return "cmd/qwen3ttsinspect -require-numeric-parity"
	case "decoder12hz_runtime":
		return "cmd/qwen3ttsinspect -require-ready"
	case "cpu_generation_runtime":
		return "cmd/lfm2inspect -require-ready"
	case "nvidia_runtime":
		return "cmd/qwen3ttsinspect -require-runtime / cmd/lfm2inspect -require-runtime"
	case "streaming_runtime":
		return "cmd/qwen3ttsinspect -require-ready"
	default:
		return ""
	}
}

func runtimeBlockerDescription(key string) string {
	switch key {
	case "cpu_talker_runtime":
		return "implement the Qwen3-TTS CPU/reference Talker semantic-token path"
	case "cpu_code_predictor_runtime":
		return "implement the Qwen3-TTS CPU/reference CodePredictor acoustic-code path"
	case "decoder12hz_runtime":
		return "implement the Qwen3-TTS 12Hz decoder and WAV/PCM output path"
	case "cpu_generation_runtime":
		return "implement the LFM2 CPU/reference generation path across embedding, conv, attention, and MoE stages"
	case "nvidia_runtime":
		return "add NVIDIA acceleration after CPU/reference parity is established"
	case "streaming_runtime":
		return "add streaming execution after CPU/reference parity is established"
	default:
		return "implement this runtime/backend coverage gate"
	}
}

func printSnapshot(w interface{ Write([]byte) (int, error) }, summaries []familySummary) {
	fmt.Fprintln(w, "# Model coverage snapshot")
	fmt.Fprintln(w)
	printMarkdownSummary(w, summaries)
	roadmap := buildRuntimeRoadmap(summaries, "")
	if len(roadmap) == 0 {
		return
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "# Runtime roadmap")
	fmt.Fprintln(w)
	printRuntimeRoadmap(w, summaries, "")
}

func summariesMeetMinPercent(summaries []familySummary, minPercent float64) bool {
	for _, s := range summaries {
		if s.CoveragePercent < minPercent {
			return false
		}
	}
	return true
}

func printCSVSummary(w interface{ Write([]byte) (int, error) }, summaries []familySummary) {
	fmt.Fprintln(w, "family,status,covered,pending,coverage_percent,references_covered,references_pending,references_percent,runtime_covered,runtime_pending,runtime_percent,execution_covered,execution_pending,execution_percent,parity_covered,parity_pending,parity_percent,readiness_covered,readiness_pending,readiness_percent")
	for _, s := range summaries {
		refs := s.Categories["references"]
		runtime := s.Categories["runtime"]
		execution := s.Categories["execution"]
		parity := s.Categories["parity"]
		readiness := s.Categories["readiness"]
		fmt.Fprintf(w, "%s,%s,%d,%d,%.1f,%d,%d,%.1f,%d,%d,%.1f,%d,%d,%.1f,%d,%d,%.1f,%d,%d,%.1f\n", s.Name, s.Status, s.Covered, s.Pending, s.CoveragePercent, refs.Covered, refs.Pending, refs.Percent, runtime.Covered, runtime.Pending, runtime.Percent, execution.Covered, execution.Pending, execution.Percent, parity.Covered, parity.Pending, parity.Percent, readiness.Covered, readiness.Pending, readiness.Percent)
	}
}

func printMarkdownSummary(w interface{ Write([]byte) (int, error) }, summaries []familySummary) {
	fmt.Fprintln(w, "| family | status | covered | pending | coverage | references | runtime | execution | parity | readiness |")
	fmt.Fprintln(w, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
	for _, s := range summaries {
		refs := s.Categories["references"]
		runtime := s.Categories["runtime"]
		execution := s.Categories["execution"]
		parity := s.Categories["parity"]
		readiness := s.Categories["readiness"]
		fmt.Fprintf(w, "| %s | %s | %d | %d | %.1f%% | %d/%d (%.1f%%) | %d/%d (%.1f%%) | %d/%d (%.1f%%) | %d/%d (%.1f%%) | %d/%d (%.1f%%) |\n", s.Name, s.Status, s.Covered, s.Pending, s.CoveragePercent, refs.Covered, refs.Pending, refs.Percent, runtime.Covered, runtime.Pending, runtime.Percent, execution.Covered, execution.Pending, execution.Percent, parity.Covered, parity.Pending, parity.Percent, readiness.Covered, readiness.Pending, readiness.Percent)
	}
}

func printTextSummary(w interface{ Write([]byte) (int, error) }, s familySummary) {
	fmt.Fprintf(w, "%s: %s covered=%d pending=%d coverage=%.1f%% runtime=%v validation=%q\n", s.Name, s.Status, s.Covered, s.Pending, s.CoveragePercent, s.RuntimeGeneration, s.ValidationTarget)
	if len(s.Categories) > 0 {
		for _, name := range []string{"references", "runtime", "execution", "parity", "readiness"} {
			counts := s.Categories[name]
			fmt.Fprintf(w, "  %s: covered=%d pending=%d coverage=%.1f%%", name, counts.Covered, counts.Pending, counts.Percent)
			if len(counts.PendingKeys) > 0 {
				fmt.Fprintf(w, " pending_keys=%v", counts.PendingKeys)
			}
			fmt.Fprintln(w)
		}
	}
	if len(s.PendingKeys) > 0 {
		fmt.Fprintf(w, "  pending: %v\n", s.PendingKeys)
	}
}

func loadManifest(path string) (manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return manifest{}, err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return manifest{}, err
	}
	if m.Version <= 0 || len(m.Families) == 0 {
		return manifest{}, fmt.Errorf("invalid model coverage manifest: version=%d families=%d", m.Version, len(m.Families))
	}
	return m, nil
}

type coverageFilter struct {
	ReferencesOnly bool
	RuntimeOnly    bool
	ExecutionOnly  bool
	ParityOnly     bool
	ReadinessOnly  bool
}

func summarize(m manifest, family string, filter coverageFilter) ([]familySummary, error) {
	names := make([]string, 0, len(m.Families))
	if family != "" {
		if _, ok := m.Families[family]; !ok {
			return nil, fmt.Errorf("unknown model family %q", family)
		}
		names = append(names, family)
	} else {
		for name := range m.Families {
			names = append(names, name)
		}
		sort.Strings(names)
	}
	out := make([]familySummary, 0, len(names))
	for _, name := range names {
		fam := m.Families[name]
		s := familySummary{Name: name, Status: fam.Status, RuntimeGeneration: fam.RuntimeGeneration, ValidationTarget: fam.ValidationTarget}
		keys := make([]string, 0, len(fam.Coverage))
		for key := range fam.Coverage {
			if !filter.include(key) {
				continue
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if fam.Coverage[key] {
				s.Covered++
			} else {
				s.Pending++
				s.PendingKeys = append(s.PendingKeys, key)
			}
		}
		s.CoveragePercent = percent(s.Covered, s.Pending)
		s.Categories = summarizeCategories(fam.Coverage)
		out = append(out, s)
	}
	return out, nil
}

func summarizeCategories(coverage map[string]bool) map[string]categoryCounts {
	categories := map[string]categoryCounts{
		"references": {},
		"runtime":    {},
		"execution":  {},
		"parity":     {},
		"readiness":  {},
	}
	for key, covered := range coverage {
		for name, include := range map[string]bool{
			"references": isReferenceCoverageKey(key),
			"runtime":    isRuntimeCoverageKey(key),
			"execution":  isExecutionCoverageKey(key),
			"parity":     isParityCoverageKey(key),
			"readiness":  isReadinessCoverageKey(key),
		} {
			if !include {
				continue
			}
			counts := categories[name]
			if covered {
				counts.Covered++
			} else {
				counts.Pending++
				counts.PendingKeys = append(counts.PendingKeys, key)
			}
			categories[name] = counts
		}
	}
	for name, counts := range categories {
		sort.Strings(counts.PendingKeys)
		counts.Percent = percent(counts.Covered, counts.Pending)
		categories[name] = counts
	}
	return categories
}

func (f coverageFilter) include(key string) bool {
	if f.ReferencesOnly && !isReferenceCoverageKey(key) {
		return false
	}
	if f.RuntimeOnly && !isRuntimeCoverageKey(key) {
		return false
	}
	if f.ExecutionOnly && !isExecutionCoverageKey(key) {
		return false
	}
	if f.ParityOnly && !isParityCoverageKey(key) {
		return false
	}
	if f.ReadinessOnly && !isReadinessCoverageKey(key) {
		return false
	}
	return true
}

func percent(covered, pending int) float64 {
	total := covered + pending
	if total == 0 {
		return 100
	}
	return 100 * float64(covered) / float64(total)
}

func isReferenceCoverageKey(key string) bool {
	return strings.Contains(key, "reference") || strings.Contains(key, "fixture")
}

func isRuntimeCoverageKey(key string) bool {
	return strings.Contains(key, "runtime") || strings.Contains(key, "nvidia") || strings.Contains(key, "streaming")
}

func isExecutionCoverageKey(key string) bool {
	return strings.HasPrefix(key, "cpu_") || key == "decoder12hz_runtime" || key == "nvidia_runtime" || key == "streaming_runtime"
}

func isParityCoverageKey(key string) bool {
	return strings.Contains(key, "parity") || strings.Contains(key, "placeholder_reference")
}

func isReadinessCoverageKey(key string) bool {
	return strings.Contains(key, "readiness") || strings.Contains(key, "ready")
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "modelcoverage:", err)
	os.Exit(1)
}
