#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
static const char* fdp(int fd){ static __thread char p[256]; char l[64]; snprintf(l,sizeof l,"/proc/self/fd/%d",fd); ssize_t n=readlink(l,p,255); if(n<0)return"?"; p[n]=0; return p; }
static volatile uint8_t *ring,*aidma,*tcm; static size_t rl,al,tl;
static FILE* lg(){ static FILE*f; if(!f)f=fopen("/tmp/sample.log","w"); return f; }
static uint64_t fnv(volatile uint8_t*p,size_t n){ uint64_t h=1469598103934665603ULL; for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;} return h; }
static void dump(const char*tag,volatile uint8_t*p,size_t n,int seq){
  fprintf(lg(),"=== %s seq=%d ===\n",tag,seq);
  for(size_t i=0;i<n;i+=32){int nz=0;for(size_t j=0;j<32&&i+j<n;j++)if(p[i+j])nz=1; if(!nz)continue;
    fprintf(lg(),"%04zx: ",i);for(size_t j=0;j<32&&i+j<n;j++)fprintf(lg(),"%02x",p[i+j]);fprintf(lg(),"\n");}
  fflush(lg());
}
static void* poll_thread(void*_){
  uint64_t seen[256]; int nseen=0, dumps=0;
  while(dumps<120){
    if(ring){ uint64_t h=fnv(ring,rl); int dup=0; for(int i=0;i<nseen;i++)if(seen[i]==h){dup=1;break;}
      if(!dup && h!=fnv((volatile uint8_t*)"\0\0\0\0\0\0\0\0",8)){ if(nseen<256)seen[nseen++]=h; if(h!=1469598103934665603ULL){dump("aidma_ring",ring,rl,dumps++);} } }
    if(aidma){ uint64_t h=fnv(aidma,al>512?512:al); int dup=0; for(int i=0;i<nseen;i++)if(seen[i]==h){dup=1;break;}
      if(!dup){ if(nseen<256)seen[nseen++]=h; dump("ai_dma_reg",aidma,al>512?512:al,dumps++); } }
    if(tcm){ for(int core=0;core<8;core++){ volatile uint8_t*w=tcm+core*0x60000; uint64_t h=fnv(w,256); int dup=0; for(int i=0;i<nseen;i++)if(seen[i]==h){dup=1;break;} if(!dup){ if(nseen<256)seen[nseen++]=h; char t[32]; snprintf(t,sizeof t,"tcm_core%d",core); dump(t,w,256,dumps++);} } }
    struct timespec ts={0,1000}; nanosleep(&ts,0);
  }
  return 0;
}
void* mmap(void*a,size_t len,int prot,int flags,int fd,off_t off){
  static void*(*real)(void*,size_t,int,int,int,off_t); if(!real)real=dlsym(RTLD_NEXT,"mmap");
  void*p=real(a,len,prot,flags,fd,off); const char*pp=fd>=0?fdp(fd):"";
  if(strstr(pp,"aidma_list")){ring=p;rl=len; static pthread_t t; pthread_create(&t,0,poll_thread,0);}
  else if(strstr(pp,"ai_dma")&&!strstr(pp,"list")){aidma=p;al=len;}
  else if(strstr(pp,"tcm")&&len>=3000000){tcm=p;tl=len;}
  return p;
}
