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
	PendingKeys       []string                  `json:"pending_keys,omitempty"`
	Categories        map[string]categoryCounts `json:"categories,omitempty"`
}

type categoryCounts struct {
	Covered     int      `json:"covered"`
	Pending     int      `json:"pending"`
	PendingKeys []string `json:"pending_keys,omitempty"`
}

func main() {
	manifestPath := flag.String("manifest", "docs/model-coverage-manifest.json", "model coverage manifest path")
	family := flag.String("family", "", "optional family name to summarize, e.g. qwen3_tts or lfm2_moe")
	jsonOut := flag.Bool("json", false, "emit JSON summary")
	markdownOut := flag.Bool("markdown", false, "emit Markdown summary table")
	csvOut := flag.Bool("csv", false, "emit CSV summary rows")
	pendingOnly := flag.Bool("pending-only", false, "only print pending coverage gate names in text mode")
	failPending := flag.Bool("fail-pending", false, "exit non-zero if any selected coverage gates are pending")
	referencesOnly := flag.Bool("references-only", false, "only include reference/fixture coverage gates in counts and pending output")
	runtimeOnly := flag.Bool("runtime-only", false, "only include runtime/backend coverage gates in counts and pending output")
	parityOnly := flag.Bool("parity-only", false, "only include numeric parity/reference readiness gates in counts and pending output")
	readinessOnly := flag.Bool("readiness-only", false, "only include readiness/report/gate coverage keys in counts and pending output")
	flag.Parse()
	m, err := loadManifest(*manifestPath)
	if err != nil {
		fatal(err)
	}
	summaries, err := summarize(m, *family, coverageFilter{ReferencesOnly: *referencesOnly, RuntimeOnly: *runtimeOnly, ParityOnly: *parityOnly, ReadinessOnly: *readinessOnly})
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
}

func printCSVSummary(w interface{ Write([]byte) (int, error) }, summaries []familySummary) {
	fmt.Fprintln(w, "family,status,covered,pending,references_covered,references_pending,runtime_covered,runtime_pending,parity_covered,parity_pending,readiness_covered,readiness_pending")
	for _, s := range summaries {
		refs := s.Categories["references"]
		runtime := s.Categories["runtime"]
		parity := s.Categories["parity"]
		readiness := s.Categories["readiness"]
		fmt.Fprintf(w, "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", s.Name, s.Status, s.Covered, s.Pending, refs.Covered, refs.Pending, runtime.Covered, runtime.Pending, parity.Covered, parity.Pending, readiness.Covered, readiness.Pending)
	}
}

func printMarkdownSummary(w interface{ Write([]byte) (int, error) }, summaries []familySummary) {
	fmt.Fprintln(w, "| family | status | covered | pending | references | runtime | parity | readiness |")
	fmt.Fprintln(w, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
	for _, s := range summaries {
		refs := s.Categories["references"]
		runtime := s.Categories["runtime"]
		parity := s.Categories["parity"]
		readiness := s.Categories["readiness"]
		fmt.Fprintf(w, "| %s | %s | %d | %d | %d/%d | %d/%d | %d/%d | %d/%d |\n", s.Name, s.Status, s.Covered, s.Pending, refs.Covered, refs.Pending, runtime.Covered, runtime.Pending, parity.Covered, parity.Pending, readiness.Covered, readiness.Pending)
	}
}

func printTextSummary(w interface{ Write([]byte) (int, error) }, s familySummary) {
	fmt.Fprintf(w, "%s: %s covered=%d pending=%d runtime=%v validation=%q\n", s.Name, s.Status, s.Covered, s.Pending, s.RuntimeGeneration, s.ValidationTarget)
	if len(s.Categories) > 0 {
		for _, name := range []string{"references", "runtime", "parity", "readiness"} {
			counts := s.Categories[name]
			fmt.Fprintf(w, "  %s: covered=%d pending=%d", name, counts.Covered, counts.Pending)
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
		s.Categories = summarizeCategories(fam.Coverage)
		out = append(out, s)
	}
	return out, nil
}

func summarizeCategories(coverage map[string]bool) map[string]categoryCounts {
	categories := map[string]categoryCounts{
		"references": {},
		"runtime":    {},
		"parity":     {},
		"readiness":  {},
	}
	for key, covered := range coverage {
		for name, include := range map[string]bool{
			"references": isReferenceCoverageKey(key),
			"runtime":    isRuntimeCoverageKey(key),
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
	if f.ParityOnly && !isParityCoverageKey(key) {
		return false
	}
	if f.ReadinessOnly && !isReadinessCoverageKey(key) {
		return false
	}
	return true
}

func isReferenceCoverageKey(key string) bool {
	return strings.Contains(key, "reference") || strings.Contains(key, "fixture")
}

func isRuntimeCoverageKey(key string) bool {
	return strings.Contains(key, "runtime") || strings.Contains(key, "nvidia") || strings.Contains(key, "streaming")
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
