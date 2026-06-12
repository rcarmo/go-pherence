#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
static FILE* lg(){ static FILE*f; if(!f){f=fopen("/tmp/io.log","w");} return f; }
static const char* fdpath(int fd){ static __thread char p[256]; char l[64]; snprintf(l,sizeof l,"/proc/self/fd/%d",fd); ssize_t n=readlink(l,p,sizeof p-1); if(n<0)return "?"; p[n]=0; return p; }
int ioctl(int fd, unsigned long req, ...){
  static int(*real)(int,unsigned long,...); if(!real)real=dlsym(RTLD_NEXT,"ioctl");
  va_list ap; va_start(ap,req); void*arg=va_arg(ap,void*); va_end(ap);
  const char*pp=fdpath(fd);
  if(strstr(pp,"tcm")||strstr(pp,"ai_dma")||strstr(pp,"aidma")||strstr(pp,"dma_msi")){
    unsigned dir=(req>>30)&3, sz=(req>>16)&0x3fff, ty=(req>>8)&0xff, nr=req&0xff;
    uint32_t before=0; if(arg) memcpy(&before,arg,4);
    int r=real(fd,req,arg);
    uint32_t after=0; if(arg) memcpy(&after,arg,4);
    fprintf(lg(),"ioctl %-14s dir=%u ty=0x%02x nr=%u sz=%u arg_in=0x%08x arg_out=0x%08x ret=%d\n",pp,dir,ty,nr,sz,before,after,r);
    fflush(lg());
    return r;
  }
  return real(fd,req,arg);
}
void* mmap(void*a,size_t len,int prot,int flags,int fd,off_t off){
  static void*(*real)(void*,size_t,int,int,int,off_t); if(!real)real=dlsym(RTLD_NEXT,"mmap");
  void*r=real(a,len,prot,flags,fd,off);
  const char*pp=fdpath(fd);
  if(fd>=0 && (strstr(pp,"tcm")||strstr(pp,"ai_dma")||strstr(pp,"aidma")||strstr(pp,"dma_msi")))
    fprintf(lg(),"mmap  %-14s len=%zu off=0x%lx prot=%d flags=%d -> %p\n",pp,len,(long)off,prot,flags,r),fflush(lg());
  return r;
}
