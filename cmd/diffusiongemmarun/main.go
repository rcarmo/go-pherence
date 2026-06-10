package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type report struct {
	ModelPath       string                             `json:"model_path"`
	PromptIDs       []int                              `json:"prompt_ids"`
	Options         diffusiongemma.InferenceOptions    `json:"options"`
	PromptTokens    []string                           `json:"prompt_tokens,omitempty"`
	GeneratedTokens []string                           `json:"generated_tokens,omitempty"`
	Capabilities    diffusiongemma.RuntimeCapabilities `json:"capabilities"`
	Shards          *diffusiongemma.ShardAvailability  `json:"shards,omitempty"`
	Result          *diffusiongemma.InferenceResult    `json:"result,omitempty"`
	Error           string                             `json:"error,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma model directory")
	promptCSV := flag.String("prompt-ids", "", "comma-separated already-tokenized prompt IDs")
	promptText := flag.String("prompt", "", "text prompt to tokenize with tokenizer.json")
	var messages stringList
	flag.Var(&messages, "message", "chat message as role:text; may be repeated; simplified scaffold, not full Jinja")
	exactTokensCSV := flag.String("tokens", "", "comma-separated exact tokenizer vocabulary entries (no BPE tokenization)")
	maxNew := flag.Int("max-new", 0, "maximum generated tokens")
	canvas := flag.Int("canvas", 0, "override canvas length")
	seed := flag.Int64("seed", 1, "deterministic canvas RNG seed")
	addBOS := flag.Bool("add-bos", false, "prepend BOS token from tokenizer metadata")
	enableThinking := flag.Bool("think", false, "prepend thinking control token from tokenizer metadata")
	addGenerationPrompt := flag.Bool("generation-prompt", false, "append generation prompt token when available")
	decode := flag.Bool("decode", false, "decode prompt/generated IDs through exact tokenizer vocabulary entries")
	mockToken := flag.Int("mock-token", -1, "use deterministic mock denoiser that always favors this token ID")
	useCPUDispatcher := flag.Bool("cpu-dispatcher", false, "open local text weights and attach the CPU/SIMD dispatcher scaffold")
	asJSON := flag.Bool("json", false, "emit JSON")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmarun -model PATH [-prompt-ids 1,2] [-max-new N] [-json]")
		os.Exit(2)
	}
	promptIDs, err := parseIDs(*promptCSV)
	if err != nil {
		fatal(err)
	}
	var vocab *diffusiongemma.Vocab
	var tok *tokenizer.Tokenizer
	if strings.TrimSpace(*promptText) != "" || len(messages) > 0 {
		tok, err = tokenizer.Load(*modelDir + "/tokenizer.json")
		if err != nil {
			fatal(err)
		}
	}
	if strings.TrimSpace(*promptText) != "" {
		promptIDs = append(promptIDs, tok.Encode(*promptText)...)
	}
	if strings.TrimSpace(*exactTokensCSV) != "" || *decode {
		vocab, err = diffusiongemma.LoadVocab(*modelDir)
		if err != nil {
			fatal(err)
		}
	}
	if strings.TrimSpace(*exactTokensCSV) != "" {
		ids, err := vocab.EncodeExact(splitCSV(*exactTokensCSV))
		if err != nil {
			fatal(err)
		}
		promptIDs = append(promptIDs, ids...)
	}
	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		fatal(err)
	}
	if *enableThinking && len(messages) == 0 {
		messages = append(messages, "system:")
	}
	if len(messages) > 0 {
		if m.Tokenizer == nil {
			fatal(fmt.Errorf("DiffusionGemma tokenizer metadata unavailable"))
		}
		specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
		chatMessages := make([]diffusiongemma.ChatMessage, 0, len(messages))
		for _, raw := range messages {
			role, content, ok := strings.Cut(raw, ":")
			if !ok {
				fatal(fmt.Errorf("bad -message %q, want role:text", raw))
			}
			role = strings.TrimSpace(role)
			text := role + "\n" + content
			if *enableThinking && (role == "system" || role == "developer") {
				text = role + "\n" + m.Processor.Think + "\n" + content
			}
			chatMessages = append(chatMessages, diffusiongemma.ChatMessage{Role: role, Content: tok.Encode(text)})
		}
		framed, err := diffusiongemma.BuildSimpleChatPromptIDs(chatMessages, specials, diffusiongemma.ChatPromptOptions{AddBOS: *addBOS})
		if err != nil {
			fatal(err)
		}
		promptIDs = append(promptIDs, framed.InputIDs...)
		if *addGenerationPrompt {
			if specials.BOT < 0 {
				fatal(fmt.Errorf("DiffusionGemma begin-turn token ID unavailable"))
			}
			promptIDs = append(promptIDs, specials.BOT)
			promptIDs = append(promptIDs, tok.Encode("model\n")...)
		}
	} else if *addBOS || *enableThinking || *addGenerationPrompt {
		if m.Tokenizer == nil {
			fatal(fmt.Errorf("DiffusionGemma tokenizer metadata unavailable"))
		}
		specials := m.Tokenizer.SpecialTokenIDs(m.Processor)
		framed, err := diffusiongemma.BuildPromptIDs(promptIDs, specials, diffusiongemma.PromptOptions{AddBOS: *addBOS, EnableThinking: *enableThinking, AddGenerationPrompt: *addGenerationPrompt})
		if err != nil {
			fatal(err)
		}
		promptIDs = framed.InputIDs
	}
	var denoiser diffusiongemma.Denoiser
	var weights *diffusiongemma.TextWeights
	if *mockToken >= 0 {
		denoiser = diffusiongemma.MockDenoiser{VocabSize: m.Shape.VocabSize, TokenID: *mockToken}
	}
	if *useCPUDispatcher {
		if m.Shards != nil && !m.Shards.Ready {
			missing := m.Shards.MissingShards
			if len(missing) > 5 {
				missing = missing[:5]
			}
			fatal(fmt.Errorf("DiffusionGemma weight shards not ready: present=%d/%d missing=%v", m.Shards.PresentShards, m.Shards.ExpectedShards, missing))
		}
		weights, err = diffusiongemma.OpenTextWeights(*modelDir, m.Shape)
		if err != nil {
			fatal(err)
		}
		defer weights.Close()
		denoiser, err = diffusiongemma.NewTextDenoiserWithDispatcher(m.Shape, weights, diffusiongemma.CPUDispatcher{})
		if err != nil {
			fatal(err)
		}
	}
	eng, err := diffusiongemma.NewEngineWithTextWeights(m, weights, denoiser)
	if err != nil {
		fatal(err)
	}
	opts := diffusiongemma.InferenceOptions{MaxNewTokens: *maxNew, CanvasLength: *canvas, Seed: *seed}
	out := report{ModelPath: *modelDir, PromptIDs: promptIDs, Options: opts, Capabilities: diffusiongemma.Capabilities(), Shards: m.Shards}
	if *decode {
		if tok != nil {
			out.PromptTokens = []string{tok.Decode(promptIDs)}
		} else if vocab != nil {
			out.PromptTokens = vocab.DecodeIDs(promptIDs)
		}
	}
	res, err := eng.GenerateTokenIDs(promptIDs, opts)
	if err != nil {
		out.Error = err.Error()
	} else {
		out.Result = &res
		if *decode {
			if tok != nil {
				out.GeneratedTokens = []string{tok.Decode(res.Generated)}
			} else if vocab != nil {
				out.GeneratedTokens = vocab.DecodeIDs(res.Generated)
			}
		}
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		if out.Error != "" {
			os.Exit(1)
		}
		return
	}
	fmt.Printf("DiffusionGemma run scaffold: %s\n", *modelDir)
	fmt.Printf("  prompt_ids=%v max_new=%d canvas=%d seed=%d cpu_dispatcher=%v mock_token=%d\n", promptIDs, opts.MaxNewTokens, opts.CanvasLength, opts.Seed, *useCPUDispatcher, *mockToken)
	if len(out.PromptTokens) > 0 {
		fmt.Printf("  prompt_tokens=%v\n", out.PromptTokens)
	}
	if out.Shards != nil {
		fmt.Printf("  shards_ready=%v present=%d/%d\n", out.Shards.Ready, out.Shards.PresentShards, out.Shards.ExpectedShards)
	}
	fmt.Printf("  caps: text_scaffold=%v reference_complete=%v encoder_kv=%v sliding_mask=%v rope=%v\n", out.Capabilities.TextOnlyScaffoldReady, out.Capabilities.ReferenceComplete, out.Capabilities.EncoderKVConcat, out.Capabilities.SlidingWindowMask, out.Capabilities.RoPE)
	if out.Error != "" {
		fmt.Printf("  error: %s\n", out.Error)
		os.Exit(1)
	}
	fmt.Printf("  generated=%v canvases=%d\n", res.Generated, len(res.Canvases))
	if len(out.GeneratedTokens) > 0 {
		fmt.Printf("  generated_tokens=%v\n", out.GeneratedTokens)
	}
}

func splitCSV(csv string) []string {
	if strings.TrimSpace(csv) == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseIDs(csv string) ([]int, error) {
	if strings.TrimSpace(csv) == "" {
		return nil, nil
	}
	parts := strings.Split(csv, ",")
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("bad token id %q: %w", part, err)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "diffusiongemmarun:", err)
	os.Exit(1)
}
