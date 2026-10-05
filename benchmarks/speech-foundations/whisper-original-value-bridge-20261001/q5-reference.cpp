// Independent retained GGML Q5_0 decoder oracle. No model inference/device use.
#include "ggml.h"
#include <cstdio>
#include <cstdint>
#include <vector>
int main(int argc,char**argv){if(argc!=3)return 2;FILE*in=fopen(argv[1],"rb");if(!in)return 3;fseek(in,0,SEEK_END);long n=ftell(in);rewind(in);if(n<=0||n>1<<20||n%22){fclose(in);return 4;}std::vector<uint8_t>b(n);if(fread(b.data(),1,n,in)!=(size_t)n){fclose(in);return 5;}fclose(in);std::vector<float>out(n/22*32);auto*t=ggml_get_type_traits(GGML_TYPE_Q5_0);if(!t||!t->to_float)return 6;t->to_float(b.data(),out.data(),out.size());FILE*f=fopen(argv[2],"wb");if(!f)return 7;bool ok=fwrite(out.data(),4,out.size(),f)==out.size();return fclose(f)==0&&ok?0:8;}
