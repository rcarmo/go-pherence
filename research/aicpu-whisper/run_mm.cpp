#include "onnxruntime_cxx_api.h"
#include "spacemit_ort_env.h"
#include <vector>
#include <cstdio>
int main(int argc,char**argv){
  bool npu=argc>2&&!strcmp(argv[2],"npu");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"m");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
  if(npu){auto st=Ort::SessionOptionsSpaceMITEnvInit(so,{}); if(!st.IsOK())fprintf(stderr,"EP:%s\n",st.GetErrorMessage().c_str());}
  Ort::Session s(env,argv[1],so);
  auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  std::vector<int8_t> A(256*256,1); int64_t sh[2]={256,256};
  auto v=Ort::Value::CreateTensor<int8_t>(mem,A.data(),A.size(),sh,2);
  const char*in[]={"A"},*out[]={"Y"};
  auto r=s.Run(Ort::RunOptions{nullptr},in,&v,1,out,1);
  fprintf(stderr,"OK Y[0]=%d\n",r[0].GetTensorData<int32_t>()[0]);
  return 0;
}
