// F32 reference kernels for the bounded MoJev text-only branch encoder.
// Compile to embedded PTX; CUDA toolkit is not a runtime dependency.
#include <math.h>
__device__ float reduce_sum(float v) {
    __shared__ float sums[8];
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    if(lane==0) sums[warp]=v;
    __syncthreads();
    v=threadIdx.x<8?sums[threadIdx.x]:0;
    if(warp==0) for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(threadIdx.x==0) sums[0]=v;
    __syncthreads();float result=sums[0];__syncthreads();return result;
}
__device__ float sigmoid(float x){return 1.f/(1.f+expf(-x));}
// 32x64 output tile, 32-wide K panels, sixteen F32 accumulators per thread (128 threads).
extern "C" __global__ void mj_gemm(const float* a,const float* b,float* c,int m,int n,int k){
 __shared__ float sa[32][32];__shared__ float sb[32][64];
 int tx=threadIdx.x%16,ty=threadIdx.x/16;
 float acc[4][4]={};
 for(int p=0;p<k;p+=32){
  for(int i=threadIdx.x;i<1024;i+=128){int r=blockIdx.y*32+i/32,d=p+i%32;sa[i/32][i%32]=(r<m&&d<k)?a[r*k+d]:0;}
  for(int i=threadIdx.x;i<2048;i+=128){int d=p+i/64,j=blockIdx.x*64+i%64;sb[i/64][i%64]=(d<k&&j<n)?b[d*n+j]:0;}
  __syncthreads();
  #pragma unroll
  for(int d=0;d<32;d++){
   #pragma unroll
   for(int r=0;r<4;r++){
    float v=sa[ty+r*8][d];
    #pragma unroll
    for(int j=0;j<4;j++)acc[r][j]=fmaf(v,sb[d][tx+j*16],acc[r][j]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int r=0;r<4;r++){
  #pragma unroll
  for(int j=0;j<4;j++){int row=blockIdx.y*32+ty+r*8,col=blockIdx.x*64+tx+j*16;if(row<m&&col<n)c[row*n+col]=acc[r][j];}
 }
}
extern "C" __global__ void mj_norm(const float* x,const float* w,float* y,int rows,int dim,float eps,int zero) {
 int row=blockIdx.x;float sum=0;
 for(int i=threadIdx.x;i<dim;i+=256){float v=x[row*dim+i];sum+=v*v;}
 float inv=rsqrtf(reduce_sum(sum)/dim+eps);
 for(int i=threadIdx.x;i<dim;i+=256)y[row*dim+i]=x[row*dim+i]*inv*(w[i]+zero);
}
extern "C" __global__ void mj_add(float* x,const float* y,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]+=y[i];}
extern "C" __global__ void mj_silu_mul(float* x,const float* up,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]=x[i]*sigmoid(x[i])*up[i];}
// Convolution weights arrive in kernel-major order, as in the Go reference.
extern "C" __global__ void mj_conv(const float* x,const float* w,float* y,int rows){
 int i=blockIdx.x*256+threadIdx.x;if(i>=rows*6144)return;int t=i/6144,c=i%6144;float s=0;
 for(int k=0;k<4;k++){int p=t-3+k;if(p>=0)s+=x[p*6144+c]*w[k*6144+c];}
 y[i]=s*sigmoid(s);
}
extern "C" __global__ void mj_l2(float* x,int rows,float eps){
 int t=blockIdx.x/32,h=blockIdx.x%32;int base=t*6144+h*128;float v=threadIdx.x<128?x[base+threadIdx.x]:0;
 float scale=rsqrtf(reduce_sum(v*v)+eps);
 if(threadIdx.x<128)x[base+threadIdx.x]=v*scale;
}
// Each warp owns a single recurrent value row, four key elements per lane.
// No state buffer persists across launches or candidate branches.
extern "C" __global__ void mj_delta(const float* qkv,const float* alpha,const float* beta,const float* dt,const float* a,float* out,int rows){
 int h=blockIdx.x/16,v=(blockIdx.x%16)*8+(threadIdx.x>>5),lane=threadIdx.x&31;
 float state[4]={0,0,0,0};
 for(int t=0;t<rows;t++){
  float b=sigmoid(beta[t*16+h]);float av=alpha[t*16+h]+dt[h];
  float step=av>20?av:log1pf(expf(av));float decay=expf(step*a[h]);
  float k[4],q[4],pred=0;
  #pragma unroll
  for(int j=0;j<4;j++){int idx=h*128+lane+j*32;k[j]=qkv[t*6144+2048+idx];q[j]=qkv[t*6144+idx];state[j]*=decay;pred+=state[j]*k[j];}
  for(int d=16;d;d>>=1)pred+=__shfl_down_sync(0xffffffff,pred,d);
  pred=__shfl_sync(0xffffffff,pred,0);
  float delta=(qkv[t*6144+4096+h*128+v]-pred)*b;float result=0;
  #pragma unroll
  for(int j=0;j<4;j++){state[j]+=k[j]*delta;result+=state[j]*q[j];}
  for(int d=16;d;d>>=1)result+=__shfl_down_sync(0xffffffff,result,d);
  if(lane==0)out[t*2048+h*128+v]=result*0.08838834764831845f;
 }
}
extern "C" __global__ void mj_gated_norm(float* x,const float* z,const float* w,int rows,float eps){
 int base=blockIdx.x*128;float v=threadIdx.x<128?x[base+threadIdx.x]:0;
 float inv=rsqrtf(reduce_sum(v*v)/128+eps);
 if(threadIdx.x<128){float g=z[base+threadIdx.x];x[base+threadIdx.x]=v*inv*w[threadIdx.x]*g*sigmoid(g);}
}
extern "C" __global__ void mj_qk_norm_rope(float* x,const float* w,int rows,int heads,int stride,int headStride,float eps){
 int t=blockIdx.x/heads,h=blockIdx.x%heads;int lane=threadIdx.x,base=t*stride+h*headStride;
 float v=x[base+lane];float inv=rsqrtf(reduce_sum(v*v)/256+eps);
 __shared__ float norm[256];norm[lane]=v*inv*(1.f+w[lane]);__syncthreads();
 if(lane<64){int j=lane%32;float angle=t*powf(10000000.f,-(2.f*j)/64.f);float c=cosf(angle),s=sinf(angle);int other=lane<32?lane+32:lane-32;v=norm[lane]*c+(lane<32?-norm[other]:norm[other])*s;}else v=norm[lane];
 x[base+lane]=v;
}
extern "C" __global__ void mj_attention(const float* qg,const float* k,const float* v,float* out,int rows,int ns,int nq){
 int t=blockIdx.x/8,h=blockIdx.x%8,kh=h/4,d=threadIdx.x;
 int end=t<ns?ns:(t<ns+nq?ns+nq:rows);
 float q=qg[t*4096+h*512+d];float maxv=-INFINITY,denom=0,acc=0;
 for(int j=0;j<end;j++){
  float score=reduce_sum(q*k[j*512+kh*256+d])*0.0625f;
  float next=fmaxf(maxv,score);float old=expf(maxv-next),weight=expf(score-next);
  denom=denom*old+weight;acc=acc*old+weight*v[j*512+kh*256+d];maxv=next;
 }
 out[t*2048+h*256+d]=(acc/denom)*sigmoid(qg[t*4096+h*512+256+d]);
}

// Tree rows store parent, local RoPE position, node start and visible end.
// Sibling rows are excluded from convolution, recurrence and full attention.
extern "C" __global__ void mj_tree_conv(const float* x,const float* w,float* y,int rows,const int* tree){
 int i=blockIdx.x*256+threadIdx.x;if(i>=rows*6144)return;int t=i/6144,c=i%6144;float s=0;
 for(int k=0;k<4;k++){int p=t;for(int back=0;back<3-k&&p>=0;back++)p=tree[p*4];if(p>=0)s+=x[p*6144+c]*w[k*6144+c];}
 y[i]=s*sigmoid(s);
}
extern "C" __global__ void mj_tree_delta(const float* qkv,const float* alpha,const float* beta,const float* dt,const float* a,float* out,int rows,const int* tree){
 int h=blockIdx.x/16,v=(blockIdx.x%16)*8+(threadIdx.x>>5),lane=threadIdx.x&31;
 float state[4]={0,0,0,0},fork[4]={0,0,0,0};bool saved=false;
 for(int t=0;t<rows;t++){
  if(tree[t*4+2]>0 && t==tree[t*4+2]){
   if(!saved){for(int j=0;j<4;j++)fork[j]=state[j];saved=true;}
   else {for(int j=0;j<4;j++)state[j]=fork[j];}
  }
  float b=sigmoid(beta[t*16+h]);float av=alpha[t*16+h]+dt[h];
  float step=av>20?av:log1pf(expf(av));float decay=expf(step*a[h]);
  float k[4],q[4],pred=0;
  #pragma unroll
  for(int j=0;j<4;j++){int idx=h*128+lane+j*32;k[j]=qkv[t*6144+2048+idx];q[j]=qkv[t*6144+idx];state[j]*=decay;pred+=state[j]*k[j];}
  for(int d=16;d;d>>=1)pred+=__shfl_down_sync(0xffffffff,pred,d);
  pred=__shfl_sync(0xffffffff,pred,0);
  float delta=(qkv[t*6144+4096+h*128+v]-pred)*b;float result=0;
  #pragma unroll
  for(int j=0;j<4;j++){state[j]+=k[j]*delta;result+=state[j]*q[j];}
  for(int d=16;d;d>>=1)result+=__shfl_down_sync(0xffffffff,result,d);
  if(lane==0)out[t*2048+h*128+v]=result*0.08838834764831845f;
 }
}
extern "C" __global__ void mj_tree_qk_norm_rope(float* x,const float* w,int rows,int heads,int stride,int headStride,float eps,const int* tree){
 int t=blockIdx.x/heads,h=blockIdx.x%heads;int lane=threadIdx.x,base=t*stride+h*headStride;
 float v=x[base+lane];float inv=rsqrtf(reduce_sum(v*v)/256+eps);
 __shared__ float norm[256];norm[lane]=v*inv*(1.f+w[lane]);__syncthreads();
 if(lane<64){int j=lane%32;float angle=tree[t*4+1]*powf(10000000.f,-(2.f*j)/64.f);float c=cosf(angle),s=sinf(angle);int other=lane<32?lane+32:lane-32;v=norm[lane]*c+(lane<32?-norm[other]:norm[other])*s;}else v=norm[lane];
 x[base+lane]=v;
}
extern "C" __global__ void mj_tree_attention(const float* qg,const float* k,const float* v,float* out,int rows,const int* tree,int prefix){
 int t=blockIdx.x/8,h=blockIdx.x%8,kh=h/4,d=threadIdx.x;
 int start=tree[t*4+2],end=tree[t*4+3],ancestors=start>0?prefix:0;
 float q=qg[t*4096+h*512+d];float maxv=-INFINITY,denom=0,acc=0;
 for(int i=0;i<ancestors+end-start;i++){
  int j=i<ancestors?i:start+i-ancestors;
  float score=reduce_sum(q*k[j*512+kh*256+d])*0.0625f;
  float next=fmaxf(maxv,score);float old=expf(maxv-next),weight=expf(score-next);
  denom=denom*old+weight;acc=acc*old+weight*v[j*512+kh*256+d];maxv=next;
 }
 out[t*2048+h*256+d]=(acc/denom)*sigmoid(qg[t*4096+h*512+256+d]);
}
