//go:build llamacpp && cgo

package main

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lllama -lggml -lggml-base -lggml-cpu -lm -lstdc++
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <llama.h>

static void quiet_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)text; (void)user_data;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec/1e9;
}

typedef struct {
    double load_s;
    double pp_s;
    double tg_s;
    int prompt_tokens;
    int gen_tokens;
    int prompt_ids[256];
    int gen_ids[512];
    float pp_tps;
    float tg_tps;
    char text[4096];
    // per-step top-5 logits (for trace mode)
    int step_top_ids[512][5];
    float step_top_vals[512][5];
} bench_result;

static int run_llama_bridge(const char * model_path, const char * prompt, int gen_tokens, int threads, int verbose, int ignore_eos, bench_result * res, char * err, size_t errn) {
    memset(res, 0, sizeof(*res));
    err[0] = 0;
    double t0 = now_s();
    if (!verbose) llama_log_set(quiet_log, NULL);
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { snprintf(err, errn, "llama_model_load_from_file failed"); return 1; }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.no_perf = true;
    if (getenv("LLAMA_DUMP_EMBD")) { cparams.embeddings = true; }
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { snprintf(err, errn, "llama_init_from_model failed"); llama_model_free(model); return 2; }
    res->load_s = now_s() - t0;

    int max_tokens = 2048;
    llama_token * toks = (llama_token*)calloc(max_tokens, sizeof(llama_token));
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks, max_tokens, true, true);
    if (n_prompt < 0) { n_prompt = -n_prompt; }
    if (n_prompt <= 0 || n_prompt > max_tokens) { snprintf(err, errn, "tokenize failed n=%d", n_prompt); llama_free(ctx); llama_model_free(model); free(toks); return 3; }
    res->prompt_tokens = n_prompt;
    for (int i = 0; i < n_prompt && i < 256; i++) res->prompt_ids[i] = toks[i];

    double pp0 = now_s();
    struct llama_batch batch = llama_batch_get_one(toks, n_prompt);
    if (llama_decode(ctx, batch) != 0) { snprintf(err, errn, "prefill llama_decode failed"); llama_free(ctx); llama_model_free(model); free(toks); return 4; }
    res->pp_s = now_s() - pp0;
    res->pp_tps = (float)((double)n_prompt / res->pp_s);
    if (getenv("LLAMA_DUMP_EMBD")) {
        int n_embd = llama_model_n_embd(model);
        float * embd = llama_get_embeddings_ith(ctx, -1);
        fprintf(stderr, "llama_hidden[0:8]:");
        for (int i = 0; i < 8 && embd; i++) fprintf(stderr, " %.5f", embd[i]);
        fprintf(stderr, "\n");
    }

    struct llama_sampler * smpl = llama_sampler_init_greedy();
    llama_token * gen = (llama_token*)calloc(gen_tokens > 0 ? gen_tokens : 1, sizeof(llama_token));
    double tg0 = now_s();
    for (int i=0; i<gen_tokens; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        gen[i] = tok;
        if (i < 512) res->gen_ids[i] = tok;
        // Capture top-5 logits for this step
        if (i < 512) {
            float * logits = llama_get_logits_ith(ctx, -1);
            int n_vocab = llama_vocab_n_tokens(vocab);
            // simple top-5 via partial scan
            int top_ids[5] = {0,0,0,0,0};
            float top_vals[5] = {-1e38f,-1e38f,-1e38f,-1e38f,-1e38f};
            for (int v = 0; v < n_vocab; v++) {
                float lv = logits[v];
                if (lv > top_vals[4]) {
                    top_vals[4] = lv; top_ids[4] = v;
                    // bubble up
                    for (int k = 3; k >= 0; k--) {
                        if (top_vals[k+1] > top_vals[k]) {
                            float tv = top_vals[k]; top_vals[k] = top_vals[k+1]; top_vals[k+1] = tv;
                            int ti = top_ids[k]; top_ids[k] = top_ids[k+1]; top_ids[k+1] = ti;
                        } else break;
                    }
                }
            }
            for (int k = 0; k < 5; k++) { res->step_top_ids[i][k] = top_ids[k]; res->step_top_vals[i][k] = top_vals[k]; }
        }
        if (!ignore_eos && llama_vocab_is_eog(vocab, tok)) { res->gen_tokens = i+1; break; }
        struct llama_batch b1 = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b1) != 0) { snprintf(err, errn, "decode token %d failed", i); break; }
        res->gen_tokens = i+1;
    }
    res->tg_s = now_s() - tg0;
    if (res->gen_tokens > 0) res->tg_tps = (float)((double)res->gen_tokens / res->tg_s);
    if (res->gen_tokens > 0) {
        int n = llama_detokenize(vocab, gen, res->gen_tokens, res->text, sizeof(res->text)-1, false, false);
        if (n >= 0 && n < (int)sizeof(res->text)) res->text[n] = 0;
    }

    free(gen);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    free(toks);
    llama_backend_free();
    return err[0] ? 5 : 0;
}
*/
import "C"

import (
	"flag"
	"fmt"
	"unsafe"
)

func main() {
	model := flag.String("model", "", "GGUF model")
	prompt := flag.String("prompt", "The capital of France is", "prompt")
	tokens := flag.Int("tokens", 64, "tokens")
	threads := flag.Int("threads", 8, "threads")
	verbose := flag.Bool("verbose", false, "show llama.cpp logs")
	ignoreEOS := flag.Bool("ignore-eos", false, "continue until -tokens even if EOS/EOG is sampled (benchmark mode)")
	traceIDs := flag.Bool("trace-ids", false, "print prompt and generated token IDs")
	traceLogits := flag.Bool("trace-logits", false, "print top-5 logits at each decode step")
	flag.Parse()
	if *model == "" {
		panic("-model required")
	}
	cm := C.CString(*model)
	cp := C.CString(*prompt)
	defer C.free(unsafe.Pointer(cm))
	defer C.free(unsafe.Pointer(cp))
	var res C.bench_result
	errbuf := (*C.char)(C.calloc(1, 4096))
	defer C.free(unsafe.Pointer(errbuf))
	verboseInt := 0
	if *verbose {
		verboseInt = 1
	}
	ignoreEOSInt := 0
	if *ignoreEOS {
		ignoreEOSInt = 1
	}
	rc := C.run_llama_bridge(cm, cp, C.int(*tokens), C.int(*threads), C.int(verboseInt), C.int(ignoreEOSInt), &res, errbuf, 4096)
	if rc != 0 {
		panic(fmt.Sprintf("rc=%d err=%s", int(rc), C.GoString(errbuf)))
	}
	fmt.Printf("load: %.3fs\n", float64(res.load_s))
	fmt.Printf("prompt tokens: %d\n", int(res.prompt_tokens))
	fmt.Printf("prefill: %.3fs %.2f tok/s\n", float64(res.pp_s), float64(res.pp_tps))
	fmt.Printf("decode: %.3fs %.2f tok/s (%d tokens)\n", float64(res.tg_s), float64(res.tg_tps), int(res.gen_tokens))
	if *traceIDs {
		fmt.Printf("prompt ids:")
		for i := 0; i < int(res.prompt_tokens) && i < 256; i++ {
			fmt.Printf(" %d", int(res.prompt_ids[i]))
		}
		fmt.Printf("\n")
		fmt.Printf("gen ids:")
		for i := 0; i < int(res.gen_tokens) && i < 512; i++ {
			fmt.Printf(" %d", int(res.gen_ids[i]))
		}
		fmt.Printf("\n")
	}
	if *traceLogits {
		for i := 0; i < int(res.gen_tokens) && i < 512; i++ {
			fmt.Printf("step %d top5:", i)
			for k := 0; k < 5; k++ {
				fmt.Printf(" %d:%.4f", int(res.step_top_ids[i][k]), float32(res.step_top_vals[i][k]))
			}
			fmt.Printf("\n")
		}
	}
	fmt.Printf("output: %s\n", C.GoString(&res.text[0]))
}
