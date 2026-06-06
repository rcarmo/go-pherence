#include "onnxruntime_cxx_api.h"
#include "spacemit_ort_env.h"
#include <vector>
#include <cstdio>
#include <cstring>
int main(int argc,char**argv){
  bool npu=argc>2&&!strcmp(argv[2],"npu");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"m");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
  if(npu){auto st=Ort::SessionOptionsSpaceMITEnvInit(so,{}); if(!st.IsOK())fprintf(stderr,"EP:%s\n",st.GetErrorMessage().c_str());}
  Ort::Session s(env,argv[1],so);
  auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions al;
  auto inName=s.GetInputNameAllocated(0,al); auto outName=s.GetOutputNameAllocated(0,al);
  std::vector<float> A(1*1500*1280); for(size_t i=0;i<A.size();i++)A[i]=((i*7)%255-128)/64.0f;
  int64_t sh[3]={1,1500,1280};
  auto v=Ort::Value::CreateTensor<float>(mem,A.data(),A.size(),sh,3);
  const char* in[]={inName.get()}; const char* out[]={outName.get()};
  for(int rep=0;rep<(argc>3?atoi(argv[3]):1);rep++){
    auto r=s.Run(Ort::RunOptions{nullptr},in,&v,1,out,1);
    auto info=r[0].GetTensorTypeAndShapeInfo();
    fprintf(stderr,"OK out dims=%zu first=%d\n",info.GetShape().size(), r[0].GetTensorData<int32_t>()[0]);
  }
  return 0;
}
