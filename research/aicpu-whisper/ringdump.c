#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
static FILE* lg(){ static FILE*f; if(!f)f=fopen("/tmp/ring.log","w"); return f; }
static const char* fdp(int fd){ static __thread char p[256]; char l[64]; snprintf(l,sizeof l,"/proc/self/fd/%d",fd); ssize_t n=readlink(l,p,255); if(n<0)return"?"; p[n]=0; return p; }
static volatile uint8_t* ring; static size_t ringlen;
static int doorbells=0;
void* mmap(void*a,size_t len,int prot,int flags,int fd,off_t off){
  static void*(*real)(void*,size_t,int,int,int,off_t); if(!real)real=dlsym(RTLD_NEXT,"mmap");
  void*r=real(a,len,prot,flags,fd,off);
  if(fd>=0 && strstr(fdp(fd),"aidma_list")){ ring=(uint8_t*)r; ringlen=len; }
  return r;
}
static void dumphex(const char*tag,volatile uint8_t*p,size_t n){
  fprintf(lg(),"--- %s (%zu bytes) ---\n",tag,n);
  for(size_t i=0;i<n;i+=32){ fprintf(lg(),"%04zx: ",i); for(size_t j=0;j<32&&i+j<n;j++) fprintf(lg(),"%02x",p[i+j]); fprintf(lg(),"\n"); }
  fflush(lg());
}
int ioctl(int fd, unsigned long req, ...){
  static int(*real)(int,unsigned long,...); if(!real)real=dlsym(RTLD_NEXT,"ioctl");
  va_list ap; va_start(ap,req); void*arg=va_arg(ap,void*); va_end(ap);
  if(strstr(fdp(fd),"dma_msi") && ring && doorbells<3){
    char t[32]; snprintf(t,sizeof t,"ring@doorbell#%d",doorbells);
    dumphex(t,ring,256); doorbells++;
  }
  return real(fd,req,arg);
}
