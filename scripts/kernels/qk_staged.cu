// Development-only dp4a subgroup kernels. Production loads embedded PTX.
// Q5 stages 512 columns per barrier cycle; Q6 retains its 256-column layout.
// Tail guards admit all GGUF K-matrix widths (multiples of 256). The four
// subgroup accumulators still visit groups in the same order.
#include <cuda_runtime.h>

template<int KIND,int J,int O,int CHUNK>
__device__ __forceinline__ void staged(const signed char *x,const float *xd,const int *xs,
 const signed char *w,const float *ws,const float *wm,float *y,int k,int n,int batch) {
 constexpr int G=KIND==5?32:16, GROUPS=CHUNK/G, THREADS=O*4;
 __shared__ __align__(16) signed char X[J][CHUNK];
 __shared__ float D[J][GROUPS];
 __shared__ int S[J][GROUPS];
 int t=threadIdx.x, sub=t&3, row=blockIdx.x*O+t/4, first=blockIdx.y*J;
 int groups=k/G;
 float acc[J]={};
 for(int base=0;base<k;base+=CHUNK){
  for(int at=t;at<J*CHUNK;at+=THREADS){int b=at/CHUNK,c=at%CHUNK;X[b][c]=first+b<batch&&(CHUNK==256||base+c<k)?x[(first+b)*k+base+c]:0;}
  for(int at=t;at<J*GROUPS;at+=THREADS){int b=at/GROUPS,g=at%GROUPS;D[b][g]=first+b<batch&&(CHUNK==256||base/G+g<groups)?xd[(first+b)*groups+base/G+g]:0;if(KIND==5)S[b][g]=first+b<batch&&(CHUNK==256||base/G+g<groups)?xs[(first+b)*groups+base/G+g]:0;}
  __syncthreads();
  if(row<n){
   #pragma unroll
   for(int g=sub;g<GROUPS;g+=4){
    int group=base/G+g;
    if(CHUNK>256 && group>=groups) continue;
    const int *wr=reinterpret_cast<const int *>(w+row*k+base+g*G);
    int weights[G/4];
    #pragma unroll
    for(int v=0;v<G/4;v++)weights[v]=wr[v];
    float scale=ws[row*groups+group];float minimum=0;if(KIND==5)minimum=wm[row*groups+group];
    #pragma unroll
    for(int j=0;j<J;j++){
     const int *xr=reinterpret_cast<const int *>(&X[j][g*G]);
     int dot=0;
     #pragma unroll
     for(int v=0;v<G/4;v++)dot=__dp4a(weights[v],xr[v],dot);
     // Permit the same multiply/subtract contraction as the existing MMQ
     // kernel. Explicit intermediate rounding changes multi-layer logits.
     float value=scale*float(dot);
     if(KIND==5)value=value-minimum*float(S[j][g]);
     acc[j]=fmaf(D[j][g],value,acc[j]);
    }
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int j=0;j<J;j++){
  float sum=acc[j];sum+=__shfl_down_sync(0xffffffff,sum,2);sum+=__shfl_down_sync(0xffffffff,sum,1);
  if(sub==0&&row<n&&first+j<batch)y[(first+j)*n+row]=sum;
 }
}
#define Q5(J,O) extern "C" __global__ __launch_bounds__(O*4) void q5_staged_j##J##_o##O(const signed char*x,const float*xd,const int*xs,const signed char*w,const float*ws,const float*wm,float*y,int k,int n,int b){staged<5,J,O,512>(x,xd,xs,w,ws,wm,y,k,n,b);}
#define Q6(J,O) extern "C" __global__ __launch_bounds__(O*4) void q6_staged_j##J##_o##O(const signed char*x,const float*xd,const signed char*w,const float*ws,float*y,int k,int n,int b){staged<6,J,O,256>(x,xd,nullptr,w,ws,nullptr,y,k,n,b);}
Q5(24,64) Q5(32,64)
Q6(24,64)
