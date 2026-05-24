//go:build spacemitort && cgo && linux

// Package spacemitort wraps SpacemiT's ONNX Runtime Execution Provider.
package spacemitort

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -lonnxruntime -lspacemit_ep
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include <spacemit_ort_env_c_api.h>

static const OrtApi* gp_ort_api() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }
static OrtStatus* gp_create_env(const OrtApi* api, OrtEnv** env) {
    return api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "go-pherence-spacemit", env);
}
static OrtStatus* gp_create_opts(const OrtApi* api, OrtSessionOptions** opt) {
    return api->CreateSessionOptions(opt);
}
static void gp_set_intra(const OrtApi* api, OrtSessionOptions* opt, int n) { api->SetIntraOpNumThreads(opt, n); }
static OrtStatus* gp_spacemit_init(OrtSessionOptions* opt, const char** keys, const char** vals, size_t n) {
    return OrtSessionOptionsSpaceMITEnvInit(opt, keys, vals, n);
}
static OrtStatus* gp_create_session(const OrtApi* api, OrtEnv* env, const char* path, OrtSessionOptions* opt, OrtSession** sess) {
    return api->CreateSession(env, path, opt, sess);
}
static OrtStatus* gp_create_cpu_meminfo(const OrtApi* api, OrtMemoryInfo** mem) {
    return api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, mem);
}
static OrtStatus* gp_create_tensor_f32(const OrtApi* api, OrtMemoryInfo* mem, float* data, size_t elem_count, int64_t* shape, size_t ndim, OrtValue** out) {
    return api->CreateTensorWithDataAsOrtValue(mem, data, elem_count * sizeof(float), shape, ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out);
}
static OrtStatus* gp_run1(const OrtApi* api, OrtSession* sess, const char* in_name, OrtValue* in_val, const char* out_name, OrtValue** out_val) {
    const char* in_names[1] = { in_name };
    const OrtValue* in_vals[1] = { in_val };
    const char* out_names[1] = { out_name };
    return api->Run(sess, NULL, in_names, in_vals, 1, out_names, 1, out_val);
}
static OrtStatus* gp_tensor_data(const OrtApi* api, OrtValue* val, float** data) {
    return api->GetTensorMutableData(val, (void**)data);
}
static const char* gp_error(const OrtApi* api, OrtStatus* st) { return api->GetErrorMessage(st); }
static void gp_release_status(const OrtApi* api, OrtStatus* st) { api->ReleaseStatus(st); }
static void gp_release_value(const OrtApi* api, OrtValue* v) { api->ReleaseValue(v); }
static void gp_release_meminfo(const OrtApi* api, OrtMemoryInfo* m) { api->ReleaseMemoryInfo(m); }
static void gp_release_session(const OrtApi* api, OrtSession* s) { api->ReleaseSession(s); }
static void gp_release_opts(const OrtApi* api, OrtSessionOptions* o) { api->ReleaseSessionOptions(o); }
static void gp_release_env(const OrtApi* api, OrtEnv* e) { api->ReleaseEnv(e); }
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const DefaultProviderLibrary = "/usr/lib/libspacemit_ep.so.2.0.2+rc5"

type Session struct {
	api  *C.OrtApi
	env  *C.OrtEnv
	opt  *C.OrtSessionOptions
	sess *C.OrtSession
	mem  *C.OrtMemoryInfo
}

func statusError(api *C.OrtApi, st *C.OrtStatus, where string) error {
	if st == nil {
		return nil
	}
	msg := C.GoString(C.gp_error(api, st))
	C.gp_release_status(api, st)
	return fmt.Errorf("%s: %s", where, msg)
}

func NewSession(modelPath string, opts Options) (*Session, error) {
	api := C.gp_ort_api()
	var env *C.OrtEnv
	if err := statusError(api, C.gp_create_env(api, &env), "CreateEnv"); err != nil {
		return nil, err
	}
	var so *C.OrtSessionOptions
	if err := statusError(api, C.gp_create_opts(api, &so), "CreateSessionOptions"); err != nil {
		C.gp_release_env(api, env)
		return nil, err
	}
	if opts.IntraThreadNum > 0 {
		C.gp_set_intra(api, so, C.int(opts.IntraThreadNum))
	}

	lib := DefaultProviderLibrary
	factory := SharedProviderEntryPoint
	kv := map[string]string{"shared_lib_path": lib, "provider_factory_entry_point": factory}
	for k, v := range opts.ProviderOptions() {
		kv[k] = v
	}
	keys := make([]*C.char, 0, len(kv))
	vals := make([]*C.char, 0, len(kv))
	for k, v := range kv {
		keys = append(keys, C.CString(k))
		vals = append(vals, C.CString(v))
	}
	defer func() {
		for _, p := range keys {
			C.free(unsafe.Pointer(p))
		}
		for _, p := range vals {
			C.free(unsafe.Pointer(p))
		}
	}()
	if err := statusError(api, C.gp_spacemit_init(so, (**C.char)(unsafe.Pointer(&keys[0])), (**C.char)(unsafe.Pointer(&vals[0])), C.size_t(len(keys))), "OrtSessionOptionsSpaceMITEnvInit"); err != nil {
		C.gp_release_opts(api, so)
		C.gp_release_env(api, env)
		return nil, err
	}

	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	var sess *C.OrtSession
	if err := statusError(api, C.gp_create_session(api, env, cpath, so, &sess), "CreateSession"); err != nil {
		C.gp_release_opts(api, so)
		C.gp_release_env(api, env)
		return nil, err
	}
	var mem *C.OrtMemoryInfo
	if err := statusError(api, C.gp_create_cpu_meminfo(api, &mem), "CreateCpuMemoryInfo"); err != nil {
		C.gp_release_session(api, sess)
		C.gp_release_opts(api, so)
		C.gp_release_env(api, env)
		return nil, err
	}
	return &Session{api: api, env: env, opt: so, sess: sess, mem: mem}, nil
}

func (s *Session) Close() {
	if s == nil || s.api == nil {
		return
	}
	if s.mem != nil {
		C.gp_release_meminfo(s.api, s.mem)
		s.mem = nil
	}
	if s.sess != nil {
		C.gp_release_session(s.api, s.sess)
		s.sess = nil
	}
	if s.opt != nil {
		C.gp_release_opts(s.api, s.opt)
		s.opt = nil
	}
	if s.env != nil {
		C.gp_release_env(s.api, s.env)
		s.env = nil
	}
}

func (s *Session) Run1(inputName string, input []float32, shape []int64, outputName string, outputElems int) ([]float32, error) {
	if len(input) == 0 || len(shape) == 0 || outputElems <= 0 {
		return nil, fmt.Errorf("invalid input/output sizes")
	}
	cInName := C.CString(inputName)
	defer C.free(unsafe.Pointer(cInName))
	cOutName := C.CString(outputName)
	defer C.free(unsafe.Pointer(cOutName))
	cshape := make([]C.int64_t, len(shape))
	for i, v := range shape {
		cshape[i] = C.int64_t(v)
	}
	var inVal *C.OrtValue
	if err := statusError(s.api, C.gp_create_tensor_f32(s.api, s.mem, (*C.float)(unsafe.Pointer(&input[0])), C.size_t(len(input)), (*C.int64_t)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), &inVal), "CreateTensor"); err != nil {
		return nil, err
	}
	defer C.gp_release_value(s.api, inVal)
	var outVal *C.OrtValue
	if err := statusError(s.api, C.gp_run1(s.api, s.sess, cInName, inVal, cOutName, &outVal), "Run"); err != nil {
		return nil, err
	}
	defer C.gp_release_value(s.api, outVal)
	var ptr *C.float
	if err := statusError(s.api, C.gp_tensor_data(s.api, outVal, &ptr), "GetTensorMutableData"); err != nil {
		return nil, err
	}
	out := make([]float32, outputElems)
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), outputElems)
	copy(out, src)
	return out, nil
}
