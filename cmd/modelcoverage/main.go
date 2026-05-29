package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
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
	Name              string   `json:"name"`
	Status            string   `json:"status"`
	RuntimeGeneration bool     `json:"runtime_generation"`
	ValidationTarget  string   `json:"validation_target"`
	Covered           int      `json:"covered"`
	Pending           int      `json:"pending"`
	PendingKeys       []string `json:"pending_keys,omitempty"`
}

func main() {
	manifestPath := flag.String("manifest", "docs/model-coverage-manifest.json", "model coverage manifest path")
	family := flag.String("family", "", "optional family name to summarize, e.g. qwen3_tts or lfm2_moe")
	jsonOut := flag.Bool("json", false, "emit JSON summary")
	pendingOnly := flag.Bool("pending-only", false, "only print pending coverage gate names in text mode")
	failPending := flag.Bool("fail-pending", false, "exit non-zero if any selected coverage gates are pending")
	flag.Parse()
	m, err := loadManifest(*manifestPath)
	if err != nil {
		fatal(err)
	}
	summaries, err := summarize(m, *family)
	if err != nil {
		fatal(err)
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summaries); err != nil {
			fatal(err)
		}
	} else {
		for _, s := range summaries {
			if *pendingOnly {
				for _, key := range s.PendingKeys {
					fmt.Printf("%s.%s\n", s.Name, key)
				}
				continue
			}
			fmt.Printf("%s: %s covered=%d pending=%d runtime=%v validation=%q\n", s.Name, s.Status, s.Covered, s.Pending, s.RuntimeGeneration, s.ValidationTarget)
			if len(s.PendingKeys) > 0 {
				fmt.Printf("  pending: %v\n", s.PendingKeys)
			}
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

func summarize(m manifest, family string) ([]familySummary, error) {
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
		out = append(out, s)
	}
	return out, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "modelcoverage:", err)
	os.Exit(1)
}
