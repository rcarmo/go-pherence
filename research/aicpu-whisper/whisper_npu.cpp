// Whisper large-v3 ONNX runner with optional SpaceMIT NPU EP.
// Usage: whisper_npu <encoder.onnx> <decoder.onnx> <mel.bin> [enc_ep] [dec_ep]
//   enc_ep/dec_ep: "npu" or "cpu" (default cpu)
// mel.bin: float32 [1,128,3000]. Emits decoded token ids (space separated) to stdout.
#include "onnxruntime_cxx_api.h"
#include "spacemit_ort_env.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

static const int SOT=50258, EN=50259, TRANSCRIBE=50360, NOTS=50364, EOS=50257;
static const int NL=32, NH=20, HD=64, DMODEL=1280, ENCLEN=1500, VOCAB=51866;

using Clock=std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

static std::vector<float> read_bin(const char* p){
  std::ifstream f(p, std::ios::binary|std::ios::ate);
  size_t n=f.tellg(); f.seekg(0); std::vector<float> v(n/4); f.read((char*)v.data(), n); return v;
}

int main(int argc, char** argv){
  try {
  if(argc<4){ fprintf(stderr,"usage:\n  %s enc.onnx dec.onnx mel.bin [enc_ep] [dec_ep]   (full)\n  %s --enc enc.onnx mel.bin out.H [ep]            (encoder only)\n  %s --dec dec.onnx in.H [ep]                     (decode only)\n",argv[0],argv[0],argv[0]); return 1; }

  // ---- encoder-only mode ----
  if(!strcmp(argv[1],"--enc")){
    bool npu = argc>5 && !strcmp(argv[5],"npu");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"e");
    Ort::SessionOptions so; so.SetIntraOpNumThreads(6);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.DisableMemPattern(); so.DisableCpuMemArena();
    if(npu){ auto st=Ort::SessionOptionsSpaceMITEnvInit(so,{}); if(!st.IsOK()) fprintf(stderr,"EP init: %s\n",st.GetErrorMessage().c_str()); }
    fprintf(stderr,"loading encoder %s (%s)...\n", argv[2], npu?"npu":"cpu");
    Ort::Session enc(env, argv[2], so);
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> mel=read_bin(argv[3]);
    int64_t mshape[3]={1,128,3000};
    std::vector<Ort::Value> ein; ein.push_back(Ort::Value::CreateTensor<float>(mem,mel.data(),mel.size(),mshape,3));
    const char* einN[]={"input_features"}; const char* eoutN[]={"last_hidden_state"};
    auto t0=Clock::now();
    auto eout=enc.Run(Ort::RunOptions{nullptr}, einN, ein.data(),1, eoutN,1);
    auto t1=Clock::now();
    auto sh=eout[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t n=sh[0]*sh[1]*sh[2];
    fprintf(stderr,"encoder %.0fms  H=[%ld,%ld,%ld]\n", ms(t0,t1), sh[0],sh[1],sh[2]);
    std::ofstream of(argv[4], std::ios::binary);
    of.write((const char*)eout[0].GetTensorData<float>(), n*4);
    fprintf(stderr,"wrote %s (%zu floats)\n", argv[4], n);
    return 0;
  }

  bool fullMode = strcmp(argv[1],"--dec")!=0;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"w");
  auto mkopts=[&](bool npu){
    Ort::SessionOptions so; so.SetIntraOpNumThreads(6);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.DisableMemPattern();
    so.DisableCpuMemArena();
    if(npu){ auto st=Ort::SessionOptionsSpaceMITEnvInit(so,{}); if(!st.IsOK()){fprintf(stderr,"EP init: %s\n",st.GetErrorMessage().c_str());} }
    return so;
  };
  auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<float> Hbuf;
  float* H=nullptr; std::vector<int64_t> Hsh={1,ENCLEN,DMODEL}; size_t Hn=(size_t)ENCLEN*DMODEL;
  std::vector<Ort::Value> eoutHolder;
  const char* decPath=argv[2]; bool decNpu=false;

  if(fullMode){
    bool encNpu = argc>4 && !strcmp(argv[4],"npu");
    decNpu = argc>5 && !strcmp(argv[5],"npu");
    Ort::SessionOptions eso=mkopts(encNpu);
    fprintf(stderr,"loading encoder (%s)...\n", encNpu?"npu":"cpu");
    Ort::Session enc(env, argv[1], eso);
    std::vector<float> mel=read_bin(argv[3]);
    int64_t mshape[3]={1,128,3000};
    std::vector<Ort::Value> ein; ein.push_back(Ort::Value::CreateTensor<float>(mem,mel.data(),mel.size(),mshape,3));
    const char* einN[]={"input_features"}; const char* eoutN[]={"last_hidden_state"};
    auto t0=Clock::now();
    eoutHolder=enc.Run(Ort::RunOptions{nullptr}, einN, ein.data(),1, eoutN,1);
    auto t1=Clock::now();
    H=eoutHolder[0].GetTensorMutableData<float>();
    Hsh=eoutHolder[0].GetTensorTypeAndShapeInfo().GetShape();
    Hn=Hsh[0]*Hsh[1]*Hsh[2];
    fprintf(stderr,"encoder %.0fms  H=[%ld,%ld,%ld]\n", ms(t0,t1), Hsh[0],Hsh[1],Hsh[2]);
  } else {
    decNpu = argc>4 && !strcmp(argv[4],"npu");
    Hbuf=read_bin(argv[3]); H=Hbuf.data(); Hn=Hbuf.size();
    Hsh={1,(int64_t)(Hn/DMODEL),DMODEL};
    fprintf(stderr,"loaded H from %s (%zu floats)\n", argv[3], Hn);
  }
  Ort::SessionOptions dso=mkopts(decNpu);
  fprintf(stderr,"loading decoder %s (%s)...\n", decPath, decNpu?"npu":"cpu");
  Ort::Session dec(env, decPath, dso);

  // ---- decoder KV-cached loop ----
  // input names: input_ids, encoder_hidden_states, use_cache_branch, + 4*NL past
  std::vector<std::string> kinds={"decoder.key","decoder.value","encoder.key","encoder.value"};
  std::vector<std::string> inNamesS={"input_ids","encoder_hidden_states","use_cache_branch"};
  for(int l=0;l<NL;l++) for(auto&k:kinds) inNamesS.push_back("past_key_values."+std::to_string(l)+"."+k);
  std::vector<const char*> inNames; for(auto&s:inNamesS) inNames.push_back(s.c_str());

  // output names: prefill requests logits + all present; steps request logits + decoder present only
  std::vector<std::string> outAllS={"logits"};
  for(int l=0;l<NL;l++) for(auto&k:kinds) outAllS.push_back("present."+std::to_string(l)+"."+k);
  std::vector<std::string> outStepS={"logits"};
  for(int l=0;l<NL;l++){ outStepS.push_back("present."+std::to_string(l)+".decoder.key"); outStepS.push_back("present."+std::to_string(l)+".decoder.value"); }
  std::vector<const char*> outAll,outStep; for(auto&s:outAllS)outAll.push_back(s.c_str()); for(auto&s:outStepS)outStep.push_back(s.c_str());

  // persistent KV storage: decoder KV grows; encoder KV fixed after prefill.
  std::vector<std::vector<float>> decK(NL),decV(NL),encK(NL),encV(NL);
  std::vector<int64_t> decLen(NL,0);

  auto buildFeeds=[&](std::vector<int64_t>& ids, bool useCache, std::vector<Ort::Value>& vals){
    vals.clear();
    int64_t idsh[2]={1,(int64_t)ids.size()};
    vals.push_back(Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), idsh, 2));
    int64_t hsh[3]={Hsh[0],Hsh[1],Hsh[2]};
    vals.push_back(Ort::Value::CreateTensor<float>(mem, H, Hn, hsh, 3));
    static bool ucT=true, ucF=false;
    int64_t ucsh[1]={1};
    vals.push_back(Ort::Value::CreateTensor<bool>(mem, useCache?&ucT:&ucF, 1, ucsh, 1));
    for(int l=0;l<NL;l++){
      int64_t dl = useCache?decLen[l]:0;
      int64_t el = useCache?(int64_t)ENCLEN:0;
      int64_t dsh[4]={1,NH,dl,HD}, esh[4]={1,NH,el,HD};
      // need non-null pointers even for 0-len
      static float dummy=0;
      vals.push_back(Ort::Value::CreateTensor<float>(mem, dl?decK[l].data():&dummy, dl*NH*HD, dsh,4));
      vals.push_back(Ort::Value::CreateTensor<float>(mem, dl?decV[l].data():&dummy, dl*NH*HD, dsh,4));
      vals.push_back(Ort::Value::CreateTensor<float>(mem, el?encK[l].data():&dummy, el*NH*HD, esh,4));
      vals.push_back(Ort::Value::CreateTensor<float>(mem, el?encV[l].data():&dummy, el*NH*HD, esh,4));
    }
  };
  auto argmaxLast=[&](Ort::Value& logits)->int{
    auto sh=logits.GetTensorTypeAndShapeInfo().GetShape(); // [1,T,VOCAB]
    int64_t T=sh[1]; const float* p=logits.GetTensorData<float>()+(T-1)*VOCAB;
    int best=0; float bv=p[0]; for(int i=1;i<VOCAB;i++) if(p[i]>bv){bv=p[i];best=i;} return best;
  };

  std::vector<int> outToks;
  auto t2=Clock::now();
  // prefill
  std::vector<int64_t> ids={SOT,EN,TRANSCRIBE,NOTS};
  std::vector<Ort::Value> vin; buildFeeds(ids,false,vin);
  auto r=dec.Run(Ort::RunOptions{nullptr}, inNames.data(), vin.data(), inNames.size(), outAll.data(), outAll.size());
  int nxt=argmaxLast(r[0]);
  // store present: indices 1.. in kinds order per layer
  for(int l=0;l<NL;l++){
    auto cp=[&](Ort::Value& v, std::vector<float>& dst, int64_t& lenOut){
      auto sh=v.GetTensorTypeAndShapeInfo().GetShape(); int64_t L=sh[2]; size_t n=1*NH*L*HD;
      dst.assign(v.GetTensorData<float>(), v.GetTensorData<float>()+n); lenOut=L;
    };
    int base=1+l*4; int64_t tmp;
    cp(r[base+0],decK[l],decLen[l]); cp(r[base+1],decV[l],tmp);
    cp(r[base+2],encK[l],tmp);       cp(r[base+3],encV[l],tmp);
  }
  if(nxt!=EOS) outToks.push_back(nxt);
  // cached steps
  for(int step=0; step<220 && nxt!=EOS; step++){
    std::vector<int64_t> sid={nxt};
    std::vector<Ort::Value> sv; buildFeeds(sid,true,sv);
    auto sr=dec.Run(Ort::RunOptions{nullptr}, inNames.data(), sv.data(), inNames.size(), outStep.data(), outStep.size());
    nxt=argmaxLast(sr[0]);
    for(int l=0;l<NL;l++){
      auto app=[&](Ort::Value& v, std::vector<float>& dst){
        auto sh=v.GetTensorTypeAndShapeInfo().GetShape(); size_t n=1*NH*sh[2]*HD;
        dst.assign(v.GetTensorData<float>(), v.GetTensorData<float>()+n);
      };
      app(sr[1+l*2+0],decK[l]); app(sr[1+l*2+1],decV[l]); decLen[l]+=1;
    }
    if(nxt==EOS) break;
    outToks.push_back(nxt);
  }
  auto t3=Clock::now();
  fprintf(stderr,"decode %.0fms  %zu tokens\n", ms(t2,t3), outToks.size());
  for(size_t i=0;i<outToks.size();i++) printf("%d%c", outToks[i], i+1<outToks.size()?' ':'\n');
  return 0;
  } catch(const std::exception& e){ fprintf(stderr,"EXC: %s\n", e.what()); return 2; }
}
