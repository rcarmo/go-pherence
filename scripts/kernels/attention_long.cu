// Long-context counterpart of the validated short attention kernels.
// Scores live in bounded per-launch global scratch, not a fixed 2048-float
// shared array. Each block still owns one query/head; no cross-block reduction.
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ __launch_bounds__(256, 1)
void gqa_attention_long(const float* q, const float* k, const float* v,
 const float* pk, const float* pv, const float* sk, const float* sv,
 const unsigned* index, float* out, float* scores,
 int rows, int pos0, int kvlen, int window, int heads, int kvheads, int dim,
 int prefix, int stride, int seqstart, int seqlen, int mode, int rowoffset,
 int scorestride, float scale) {
 const int row=blockIdx.y+rowoffset, head=blockIdx.x;
 if(row>=rows||head>=heads)return;
 const int tid=threadIdx.x,lane=tid&31,warp=tid/32,kv=head/(heads/kvheads);
 const int qoff=(row*heads+head)*dim,kvdim=kvheads*dim;
 int start=0,parent=0,plen=0,branch=0,first=0,end=0;
 if(mode==0){end=min(pos0+row+1,kvlen);first=window>0?max(0,end-window):0;}
 else if(mode==1){branch=index[row];first=seqstart;end=first+seqlen;}
 else {start=index[row*3];parent=index[row*3+1];plen=index[row*3+2];end=prefix+plen+row-start+1;first=window>0?max(0,end-window):0;}
 const int count=end-first;
 float* s=scores+(blockIdx.y*heads+head)*scorestride;
 __shared__ float reduce[256];
 for(int t=warp;t<count;t+=8){
  const int at=first+t; const float* kr;
  if(mode==0)kr=k+at*kvdim+kv*dim;
  else if(mode==1)kr=at<prefix?pk+at*kvdim+kv*dim:sk+(branch*stride+at-prefix)*kvdim+kv*dim;
  else kr=at<prefix?pk+at*kvdim+kv*dim:k+((at<prefix+plen?parent+at-prefix:start+at-prefix-plen)*kvdim)+kv*dim;
  float dot=0;for(int d=lane;d<dim;d+=32)dot=fmaf(q[qoff+d],kr[d],dot);
  for(int off=16;off;off>>=1)dot+=__shfl_down_sync(0xffffffff,dot,off);
  if(lane==0)s[t]=dot*scale;
 }
 __syncthreads();
 float mx=-INFINITY;for(int t=tid;t<count;t+=256)mx=fmaxf(mx,s[t]);reduce[tid]=mx;__syncthreads();
 for(int off=128;off;off>>=1){if(tid<off)reduce[tid]=fmaxf(reduce[tid],reduce[tid+off]);__syncthreads();}
 mx=reduce[0];__syncthreads();
 float sum=0;for(int t=tid;t<count;t+=256){float p=expf(s[t]-mx);s[t]=p;sum+=p;}reduce[tid]=sum;__syncthreads();
 for(int off=128;off;off>>=1){if(tid<off)reduce[tid]+=reduce[tid+off];__syncthreads();}
 float inv=1.f/reduce[0];for(int t=tid;t<count;t+=256)s[t]*=inv;__syncthreads();
 for(int d=tid;d<dim;d+=256){float acc=0;for(int t=0;t<count;t++){
  const int at=first+t; const float* vr;
  if(mode==0)vr=v+at*kvdim+kv*dim;
  else if(mode==1)vr=at<prefix?pv+at*kvdim+kv*dim:sv+(branch*stride+at-prefix)*kvdim+kv*dim;
  else vr=at<prefix?pv+at*kvdim+kv*dim:v+((at<prefix+plen?parent+at-prefix:start+at-prefix-plen)*kvdim)+kv*dim;
  acc=fmaf(s[t],vr[d],acc);
 }out[qoff+d]=acc;}
}
