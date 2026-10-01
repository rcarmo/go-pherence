// Diagnostic harness includes pinned-copy source to access mel state; no production library edit.
#include "whisper-diagnostic.cpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
using Clock=std::chrono::steady_clock;
static bool abort_deadline(void*p){return std::chrono::duration<double>(Clock::now()-*static_cast<Clock::time_point*>(p)).count()>110;}
static bool write_float(const std::string&path,const std::vector<float>&v){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(v.data()),v.size()*4);return bool(out);}
int main(int argc,char**argv){if(argc!=4)return 2;auto start=Clock::now();const std::string prefix=argv[3];std::ifstream in(argv[2],std::ios::binary|std::ios::ate);auto size=in.tellg();if(size<=0||size>16000*60*4||size%4)return 3;std::vector<float> pcm(size_t(size)/4);in.seekg(0);in.read(reinterpret_cast<char*>(pcm.data()),size);if(!in)return 4;
 auto cp=whisper_context_default_params();cp.use_gpu=true;cp.flash_attn=true;auto*ctx=whisper_init_from_file_with_params(argv[1],cp);if(!ctx)return 5;if(whisper_pcm_to_mel(ctx,pcm.data(),int(pcm.size()),4)){whisper_free(ctx);return 6;}auto &mel=ctx->state->mel;const int frames=3000,bands=ctx->model.hparams.n_mels;std::vector<float> window(frames*bands);for(int m=0;m<bands;m++)for(int t=0;t<frames;t++)if(t<mel.n_len)window[m*frames+t]=mel.data[m*mel.n_len+t];if(!write_float(prefix+"-mel.f32",window)){whisper_free(ctx);return 7;}
 std::ofstream meta(prefix+"-mel.json");meta<<"{\"frames\":3000,\"bands\":"<<bands<<",\"original_mel_length\":"<<mel.n_len<<",\"original_audio_frames\":"<<mel.n_len_org<<",\"samples\":"<<pcm.size()<<",\"hidden_layout\":\"contiguous D1280 by T1500 (Go time-row-major)\"}\n";meta.close();
 const auto hidden=prefix+"-hidden.f32";setenv("WHISPER_DIAGNOSTIC_HIDDEN_OUT",hidden.c_str(),1);if(whisper_encode(ctx,0,4)){whisper_free(ctx);return 8;}unsetenv("WHISPER_DIAGNOSTIC_HIDDEN_OUT");
 std::vector<whisper_token> tokens={50258,50259,50360,50365,400,370,11,452,7177,6280,11,1029,406,437,428,1941,393,360,337,291,11,1029,437,291,393,360,337,428,1941,13,50868};std::ofstream scores(prefix+"-logits.json");scores<<std::setprecision(12)<<"[";for(size_t i=0;i<tokens.size();i++){if(abort_deadline(&start)){whisper_free(ctx);return 9;}if(whisper_decode(ctx,&tokens[i],1,int(i),4)){whisper_free(ctx);return 10;}const int vocab=whisper_n_vocab(ctx);std::vector<float> logits(whisper_get_logits(ctx),whisper_get_logits(ctx)+vocab);if(!write_float(prefix+"-logits-"+std::to_string(i)+".f32",logits)){whisper_free(ctx);return 11;}if(i)scores<<',';scores<<"{\"position\":"<<i<<",\"input\":"<<tokens[i]<<",\"eot\":"<<logits[50257]<<",\"repeat_timestamp\":"<<logits[50868]<<"}";}
 scores<<"]\n";whisper_free(ctx);std::cerr<<"STATE_EXPORT_PREFIX_PASS tokens="<<tokens.size()<<"\n";return scores?0:12;}
