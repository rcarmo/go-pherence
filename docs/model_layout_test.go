package docs_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// This check deliberately walks the worktree, not git ls-files: new templates,
// ignored source accidentally placed under models/, and foreign-architecture
// Go files must be checked before they are staged or executed. No Git, Python,
// Bun, model weights or hardware is required by the Go gate.
func TestModelLayout(t *testing.T) {
	violations, err := auditModelLayout("..")
	if err != nil {
		t.Fatal(err)
	}
	if len(violations) != 0 {
		t.Fatalf("model source belongs in model/ and local assets in checkpoints/:\n%s", strings.Join(violations, "\n"))
	}
}

// These files record dated experiments, not current invocation defaults. Keep
// exact exceptions: a new executable in benchmarks/ must still obey the layout.
var archivedLayoutFiles = map[string]bool{
	"benchmarks/speech-foundations/final-release-verification-20260913/run-supported-checks.sh": true,
	"benchmarks/speech-foundations/sincnet-lowered-fma-20260912/filter.py":                      true,
	"benchmarks/speech-foundations/word-speaker-alignment-20260913/transformers-oracle.py":      true,
	"model/diffusiongemma/testdata/gguf_hi_1x1_parity_status.json":                              true,
	"model/diffusiongemma/testdata/llamacpp_gguf_hi_1x1_reference.json":                         true,
}

var layoutTokens = regexp.MustCompile(`[A-Za-z0-9_./${}():+@~\\-]*models[/\\][A-Za-z0-9_.*<>{}/\\-]*`)
var joinedPaths = regexp.MustCompile(`(?s)(?:filepath\.(?:Join|FromSlash)|path\.(?:join|resolve)|os\.path\.join|Path)\([^)]*\)`)
var pathLiterals = regexp.MustCompile(`["'` + "`" + `]([^"'` + "`" + `]+)["'` + "`" + `]`)
var oldDefault = regexp.MustCompile(`(?m)(?:default\s*=|(?:MODELS_DIR|CHECKPOINTS_DIR)\s*(?:\?=|:=|=))\s*["']?models(?:["'\s,)]|$)`)
var oldRootAssignment = regexp.MustCompile(`(?im)\b(?:root|assetRoot|modelRoot|modelsRoot|checkpointRoot|checkpointDir|checkpointsDir|modelsDir|modelDir|models_dir|model_dir|checkpoints_dir)["']?\s*(?::=|=|:)\s*["']models["']`)

func auditModelLayout(root string) ([]string, error) {
	var violations []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if entry.IsDir() {
			// Assets are arbitrary third-party data (sometimes remote Python code).
			// Do not traverse them, caches, tool environments or binary output.
			if rel != "." && (entry.Name() == ".git" || strings.HasPrefix(entry.Name(), ".venv") || entry.Name() == ".gotmp" || entry.Name() == ".cache" || entry.Name() == ".pytest_cache" || entry.Name() == "node_modules" || entry.Name() == "__pycache__" || rel == "checkpoints" || rel == "tmp" || rel == "bin" || rel == "logs") {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil // never follow a weight symlink out of the repository
		}
		ext := strings.ToLower(filepath.Ext(rel))
		if strings.HasPrefix(rel, "models/") {
			if ext == ".go" || ext == ".s" || ext == ".c" || ext == ".h" || strings.HasSuffix(rel, "/go.mod") || strings.HasSuffix(rel, "/Makefile") {
				violations = append(violations, rel+": source/build file in legacy asset tree")
			}
			return nil // ordinary old-checkout weights are allowed, never loaded
		}
		// Guard fixtures intentionally contain old paths to test rejection.
		if rel == "docs/model_layout_test.go" || rel == "scripts/model-layout.test.ts" || archivedLayoutFiles[rel] {
			return nil
		}
		textFile := entry.Name() == "Makefile" || entry.Name() == "Dockerfile"
		switch ext {
		case ".go", ".mod", ".work", ".py", ".ts", ".js", ".sh", ".mk", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".service", ".json", ".md", ".tmpl", ".tpl", ".gotmpl", ".template", ".in":
			textFile = true
		}
		historical := strings.HasPrefix(rel, "benchmarks/") || strings.HasPrefix(rel, "docs/history/") || strings.Contains(rel, "/experiments/")
		if !textFile || (historical && (ext == ".md" || ext == ".json")) || strings.HasSuffix(rel, "/PROVENANCE.md") || strings.HasSuffix(rel, "/VALIDATION.md") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		if ext == ".md" {
			text = layoutCodeFences(text) // migration explanations can name old paths
		}
		for _, problem := range layoutContentProblems(rel, text) {
			violations = append(violations, rel+": "+problem)
		}
		return nil
	})
	return violations, err
}

func layoutCodeFences(text string) string {
	var out strings.Builder
	fence := ""
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			if fence == "" {
				fence = trimmed[:3]
			} else if strings.HasPrefix(trimmed, fence) {
				fence = ""
			}
			continue
		}
		if fence != "" {
			out.WriteString(line + "\n")
		}
	}
	return out.String()
}

func layoutContentProblems(path, text string) []string {
	var problems []string
	if strings.Contains(text, "github.com/rcarmo/go-pherence/"+"models/") {
		problems = append(problems, "old module import path")
	}
	for _, token := range layoutTokens.FindAllString(text, -1) {
		if legacyLocalModelPath(path, token) {
			problems = append(problems, "legacy repository path "+token)
		}
	}
	for _, call := range joinedPaths.FindAllString(text, -1) {
		for _, literal := range pathLiterals.FindAllStringSubmatch(call, -1) {
			if literal[1] == "." || literal[1] == ".." {
				continue
			}
			if literal[1] == "models" {
				problems = append(problems, "legacy joined path "+call)
			}
			break // cmd/models, docs/models and sibling repos are intentional
		}
	}
	if oldDefault.MatchString(text) || oldRootAssignment.MatchString(text) {
		problems = append(problems, "legacy asset-directory default")
	}
	return problems
}

func legacyLocalModelPath(file, token string) bool {
	token = strings.ReplaceAll(token, `\`, "/")
	// Root interpolation isn't an external store. Match before absolute paths.
	for _, prefix := range []string{"$PWD/", "${PWD}/", "$(CURDIR)/", "${REPO_ROOT}/", "$REPO_ROOT/", "${ROOT}/", "$ROOT/"} {
		token = strings.TrimPrefix(token, prefix)
	}
	if idx := strings.Index(token, "/go-pherence/models/"); idx >= 0 && !strings.Contains(token, "://") {
		token = token[idx+len("/go-pherence/"):]
	}
	if strings.HasPrefix(token, "/") || strings.Contains(token, "://") {
		return false
	}
	for strings.HasPrefix(token, "../") || strings.HasPrefix(token, "./") {
		if strings.HasPrefix(token, "../") {
			token = strings.TrimPrefix(token, "../")
		} else {
			token = strings.TrimPrefix(token, "./")
		}
	}
	if !strings.HasPrefix(token, "models/") {
		return false // cmd/models, docs/models and sibling repositories
	}
	// Paths into upstream pyannote Python sources, not our checkpoints.
	if strings.HasPrefix(file, "scripts/community1_") && (strings.HasPrefix(token, "models/segmentation/") || strings.HasPrefix(token, "models/blocks/") || strings.HasPrefix(token, "models/embedding/")) {
		return false
	}
	// The inspector's roadmap path is relative to docs/, not the repository.
	if file == "docs/model_coverage_manifest_test.go" && token == "models/minicpmv-runtime-roadmap.md" {
		return false
	}
	// Existing prose uses slash-separated nouns, not filesystem paths. Scope
	// these exceptions so a new models/window default elsewhere still fails.
	prose := map[string]string{
		"runtime/speechjob/httpapi/http.go":              "models/backend",
		"model/speaker/community1/vulkan_diarization.go": "models/window",
		"scripts/community1_raw_plda_reference.py":       "models/media/GPU.",
	}
	if prose[file] == token {
		return false
	}
	return true
}

func TestModelLayoutGuardRejectsRegressions(t *testing.T) {
	badImport := "github.com/rcarmo/go-pherence/" + "models/whisper"
	for _, tt := range []struct{ name, path, content string }{
		{"import", "model/future.go", "package future\nimport _ \"" + badImport + "\""},
		{"foreign", "model/future_riscv64.go", "//go:build riscv64\npackage future\nimport _ \"" + badImport + "\""},
		{"template", "templates/model.go.tmpl", "import \"" + badImport + "\""},
		{"new-script", "scripts/new.sh", "go test ./" + "models/whisper"},
		{"benchmark-script", "benchmarks/new/check.sh", "go test ./" + "models/bert"},
		{"experiment-code", "model/new/experiments/generate.py", `output = "` + "models/new" + `"`},
		{"history-code", "docs/history/new-template.sh", "go test ./" + "models/bert"},
		{"module-template", "templates/go.mod.tmpl", "module github.com/rcarmo/go-pherence/" + "models/new"},
		{"CI", ".github/workflows/test.yml", "run: go test ./" + "models/whisper"},
		{"hidden-scaffold", ".devcontainer/devcontainer.json", `{"postCreateCommand":"go test ./` + "models/bert" + `"}`},
		{"absolute-repo", "scripts/new.sh", "/workspace/projects/go-pherence/" + "models/qwen"},
		{"legacy-ignored-source", "models/new/x.go", "package x"},
		{"fixture", "model/testdata/future.json", `{"main_model":"` + "models/checkpoint" + `"}`},
		{"root-example", "docs/guides/future.md", "```bash\nMODEL=\"$PWD/" + "models/new\"\n```"},
		{"joined", "model/future_test.go", `filepath.Join("..", ` + `"models", "weights")`},
		{"multiline-joined", "scripts/new.py", "os.path.join(\n  root, '..',\n  '" + "models', 'weights')"},
		{"windows", "templates/model.txt.in", `..\` + `models\weights`},
		{"make-root", "Makefile", "WEIGHTS = $(CURDIR)/" + "models/weights"},
		{"python-default", "scripts/new.py", `parser.add_argument("--checkpoints-dir", default=` + `"models")`},
		{"make-default", "Makefile", "CHECKPOINTS_DIR ?= " + "models\n"},
		{"literal-root", "scripts/generate.ts", `const modelRoot = "` + `models"`},
		{"json-root", "templates/config.json", `{"checkpoints_dir": "` + `models"}`},
		{"go-root", "model/new.go", `root := "` + `models"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, tt.path)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(tt.content), 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := auditModelLayout(root)
			if err != nil || len(got) == 0 {
				t.Fatalf("audit = %v, %v; want a layout violation", got, err)
			}
		})
	}
}

func TestModelLayoutGuardAllowsExternalAndHistoricalPaths(t *testing.T) {
	for _, token := range []string{"/workspace/models/shared", "/opt/models/qwen", "../llama.cpp/models/qwen", "../../../gte-go/models/gte-small", "cmd/models/inspect", "docs/models/supported.md", "https://host/models/example"} {
		if legacyLocalModelPath("scripts/example.sh", token) {
			t.Errorf("rejected external/documentation path %q", token)
		}
	}
	for _, text := range []string{
		`filepath.Join("..", "llama.cpp", "models", "weights")`,
		`path.join("cmd", "models", "inspector")`,
		`{"models": {"architecture": "trellis2"}}`,
	} {
		if got := layoutContentProblems("scripts/current.ts", text); len(got) != 0 {
			t.Errorf("rejected valid path or metadata %q: %v", text, got)
		}
	}
	root := t.TempDir()
	for _, path := range []string{"checkpoints/third-party/model.go", "models/old/model.safetensors", "docs/history/old.md", "benchmarks/run/evidence.json", "model/omnivoice/experiments/rejected.patch"} {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("models/"+"old"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if got, err := auditModelLayout(root); err != nil || len(got) != 0 {
		t.Fatalf("audit = %v, %v; want no violations", got, err)
	}
}
