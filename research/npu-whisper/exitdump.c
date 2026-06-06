#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
static volatile uint8_t *ring,*tcm,*aidma; static size_t rl,tl,al;
static const char* fdp(int fd){ static __thread char p[256]; char l[64]; snprintf(l,sizeof l,"/proc/self/fd/%d",fd); ssize_t n=readlink(l,p,255); if(n<0)return"?"; p[n]=0; return p; }
void* mmap(void*a,size_t len,int prot,int flags,int fd,off_t off){
  static void*(*r)(void*,size_t,int,int,int,off_t); if(!r)r=dlsym(RTLD_NEXT,"mmap");
  void*p=r(a,len,prot,flags,fd,off); const char*pp=fd>=0?fdp(fd):"";
  if(strstr(pp,"aidma_list")){ring=p;rl=len;}
  else if(strstr(pp,"ai_dma")){aidma=p;al=len;}
  else if(strstr(pp,"tcm")&&len>=3000000){tcm=p;tl=len;}
  return p;
}
__attribute__((destructor)) static void dump(){
  FILE*f=fopen("/tmp/exit.log","w"); if(!f)return;
  void hx(const char*t,volatile uint8_t*p,size_t n){ if(!p){fprintf(f,"%s: null\n",t);return;} fprintf(f,"=== %s (%zu) ===\n",t,n);
    for(size_t i=0;i<n;i+=32){int nz=0;for(size_t j=0;j<32&&i+j<n;j++)if(p[i+j])nz=1; if(!nz)continue; fprintf(f,"%05zx: ",i);for(size_t j=0;j<32&&i+j<n;j++)fprintf(f,"%02x",p[i+j]);fprintf(f,"\n");}}
  hx("aidma_list_ring",ring,rl);
  hx("ai_dma_region",aidma, al>4096?4096:al);
  hx("tcm_head",tcm, tl>2048?2048:tl);
  fclose(f);
}
