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
    float pp_tps;
    float tg_tps;
    char text[4096];
} bench_result;

static int run_llama_bridge(const char * model_path, const char * prompt, int gen_tokens, int threads, bench_result * res, char * err, size_t errn) {
    memset(res, 0, sizeof(*res));
    err[0] = 0;
    double t0 = now_s();
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
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { snprintf(err, errn, "llama_init_from_model failed"); llama_model_free(model); return 2; }
    res->load_s = now_s() - t0;

    int max_tokens = 2048;
    llama_token * toks = (llama_token*)calloc(max_tokens, sizeof(llama_token));
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), toks, max_tokens, true, true);
    if (n_prompt < 0) { n_prompt = -n_prompt; }
    if (n_prompt <= 0 || n_prompt > max_tokens) { snprintf(err, errn, "tokenize failed n=%d", n_prompt); llama_free(ctx); llama_model_free(model); free(toks); return 3; }
    res->prompt_tokens = n_prompt;

    double pp0 = now_s();
    struct llama_batch batch = llama_batch_get_one(toks, n_prompt);
    if (llama_decode(ctx, batch) != 0) { snprintf(err, errn, "prefill llama_decode failed"); llama_free(ctx); llama_model_free(model); free(toks); return 4; }
    res->pp_s = now_s() - pp0;
    res->pp_tps = (float)((double)n_prompt / res->pp_s);

    struct llama_sampler * smpl = llama_sampler_init_greedy();
    int text_len = 0;
    double tg0 = now_s();
    for (int i=0; i<gen_tokens; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        const char * piece = llama_vocab_get_text(vocab, tok);
        if (piece && text_len < (int)sizeof(res->text)-8) {
            int n = snprintf(res->text + text_len, sizeof(res->text)-text_len, "%s", piece);
            if (n > 0) text_len += n;
        }
        if (llama_vocab_is_eog(vocab, tok)) { res->gen_tokens = i+1; break; }
        struct llama_batch b1 = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b1) != 0) { snprintf(err, errn, "decode token %d failed", i); break; }
        res->gen_tokens = i+1;
    }
    res->tg_s = now_s() - tg0;
    if (res->gen_tokens > 0) res->tg_tps = (float)((double)res->gen_tokens / res->tg_s);

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
	rc := C.run_llama_bridge(cm, cp, C.int(*tokens), C.int(*threads), &res, errbuf, 4096)
	if rc != 0 {
		panic(fmt.Sprintf("rc=%d err=%s", int(rc), C.GoString(errbuf)))
	}
	fmt.Printf("load: %.3fs\n", float64(res.load_s))
	fmt.Printf("prompt tokens: %d\n", int(res.prompt_tokens))
	fmt.Printf("prefill: %.3fs %.2f tok/s\n", float64(res.pp_s), float64(res.pp_tps))
	fmt.Printf("decode: %.3fs %.2f tok/s (%d tokens)\n", float64(res.tg_s), float64(res.tg_tps), int(res.gen_tokens))
	fmt.Printf("output: %s\n", C.GoString(&res.text[0]))
}
