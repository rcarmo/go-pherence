// Development-only source. Production loads the generated PTX through Go.
#include <cuda_runtime.h>
#include <float.h>

extern "C" __global__ void rope_partial_segmented(float *x, const float *cs,
    const int *starts, int rows, int prefix, int heads, int dim, int rot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * rot) return;
    int row = i / (heads * rot), pair = i % rot, head = (i / rot) % heads;
    int pos = prefix + row - starts[row*3] + starts[row*3+2];
    int at = (row * heads + head) * dim + pair;
    float a=x[at], b=x[at+rot], c=cs[2*(pos*rot+pair)], s=cs[2*(pos*rot+pair)+1];
    x[at] = fmaf(a,c,b*(-s));
    x[at+rot] = fmaf(a,s,b*c);
}

// Each block reads immutable shared prefix KV and only its own causal suffix.
// No concatenated KV copy, padding, or visibility across sibling sequences.
extern "C" __global__ __launch_bounds__(256,1) void gqa_attention_segmented(
    const float *q, const float *k, const float *v, const float *pk, const float *pv,
    float *out, const int *starts, int rows, int prefix, int window,
    int heads, int kvheads, int dim, float scale) {
    int head=blockIdx.x, row=blockIdx.y, t=threadIdx.x;
    if(head>=heads || row>=rows) return;
    int lane=t&31, warp=t>>5, start=starts[row*3], parent=starts[row*3+1], plen=starts[row*3+2];
    int end=prefix+plen+row-start+1, first=window>0 && end>window ? end-window : 0;
    int count=end-first, kvhead=head/(heads/kvheads), stride=kvheads*dim;
    const float *qr=q+(row*heads+head)*dim;
    __shared__ float scores[2048], reduce[256];
    for(int j=warp;j<count;j+=8) {
        int pos=first+j;
        const float *kr=pos<prefix ? pk+pos*stride+kvhead*dim : k+(pos<prefix+plen ? parent+pos-prefix : start+pos-prefix-plen)*stride+kvhead*dim;
        float sum=0;
        for(int d=lane;d<dim;d+=32) sum=fmaf(qr[d],kr[d],sum);
        for(int n=16;n;n>>=1) sum+=__shfl_down_sync(0xffffffff,sum,n);
        if(lane==0) scores[j]=sum*scale;
    }
    __syncthreads();
    float maximum=-FLT_MAX;
    for(int j=t;j<count;j+=256) maximum=fmaxf(maximum,scores[j]);
    reduce[t]=maximum;
    __syncthreads();
    for(int n=128;n;n>>=1) { if(t<n) reduce[t]=fmaxf(reduce[t],reduce[t+n]); __syncthreads(); }
    maximum=reduce[0];
    // A warp with no scores must not overwrite the maximum before peers read it.
    __syncthreads();
    float sum=0;
    for(int j=t;j<count;j+=256) { float e=__expf(scores[j]-maximum); scores[j]=e; sum+=e; }
    reduce[t]=sum;
    __syncthreads();
    for(int n=128;n;n>>=1) { if(t<n) reduce[t]+=reduce[t+n]; __syncthreads(); }
    float inv=1.0f/reduce[0];
    for(int j=t;j<count;j+=256) scores[j]*=inv;
    __syncthreads();
    for(int d=t;d<dim;d+=256) {
        float value=0;
        for(int j=0;j<count;j++) {
            int pos=first+j;
            const float *vr=pos<prefix ? pv+pos*stride+kvhead*dim : v+(pos<prefix+plen ? parent+pos-prefix : start+pos-prefix-plen)*stride+kvhead*dim;
            value=fmaf(scores[j],vr[d],value);
        }
        out[(row*heads+head)*dim+d]=value;
    }
}

// One warp per (terminal hidden row, candidate vocabulary row) pair.
// owners maps each output to its hidden row; ids gives the vocabulary row.
extern "C" __global__ void gemv_q5_packed_selected_batch_f32(
    const float *x, const unsigned char *q, const float *scales, const float *mins,
    const unsigned *ids, const unsigned *owners, float *out, int width, int count) {
    int item=blockIdx.x, lane=threadIdx.x;
    if(item>=count) return;
    unsigned id=ids[item], owner=owners[item];
    float sum=0;
    for(int d=lane;d<width;d+=32) {
        int group=id*(width/32)+d/32;
        // Match the existing F32 selected-row dequantisation: separate mul/sub.
        float weight=__fsub_rn(__fmul_rn(scales[group],float(q[id*width+d])),mins[group]);
        sum=fmaf(x[owner*width+d],weight,sum);
    }
    for(int n=16;n;n>>=1) sum+=__shfl_down_sync(0xffffffff,sum,n);
    if(lane==0) out[item]=sum;
}
