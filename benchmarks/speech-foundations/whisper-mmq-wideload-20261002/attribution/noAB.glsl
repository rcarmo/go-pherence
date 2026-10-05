#version 450
#extension GL_EXT_integer_dot_product : require
// Q5_0 weights x Q8_1 activations, 64 features x 64 rows per 128-thread group,
// four K blocks per shared-memory stage. Per output the integer block sums and
// F32 epilogue are identical to linear.glsl: precise dot*ds.x-16*ds.y, then
// fma(d,corrected,acc) in increasing block order, bias added last.
// Requires K%128==0. Weight blocks: [d|0, qh, qs0..qs3]; activations: [ds, q0..q7].
layout(local_size_x=128) in;
layout(set=0,binding=0) readonly buffer X {uint x[];};
layout(set=0,binding=1) readonly buffer W {uint w[];};
layout(set=0,binding=2) readonly buffer B {float bias[];};
layout(set=0,binding=3) buffer Y {float y[];};
layout(push_constant) uniform P {uint M;uint K;uint N;};
const uint STEP=4;
shared uint as_[64*STEP*9];
shared uint bs_[64*STEP*9];
float corrected(int dot,vec2 q){precise float a=float(dot)*q.x;precise float b=a-16.0*q.y;return b;}
void main(){
 uint tid=gl_LocalInvocationID.x,fx=tid%16,ty=tid/16;
 uint col0=gl_WorkGroupID.x*64,row0=gl_WorkGroupID.y*64,nb=K/32;
 float acc[4][8];
 for(uint i=0;i<4;i++)for(uint j=0;j<8;j++)acc[i][j]=0.0;
 for(uint block=0;block<nb;block+=STEP){
  // A ablation: no global reads.
  for(uint item=tid;item<64*STEP;item+=128){
   uint f=item/STEP,s=item%STEP;uint dst=(s*64+f)*9;
   for(uint g=0;g<9;g++)as_[dst+g]=(item*2246822519u)+(block*3266489917u)+g;
  }
  // B ablation: no global reads.
  for(uint item=tid;item<64*STEP*9;item+=128){
   uint r=item/(STEP*9),o=item%(STEP*9),s=o/9,g=o%9;
   bs_[(s*64+r)*9+g]=(item*2654435761u)+(block*40503u);
  }
  barrier();
  for(uint s=0;s<STEP;s++){
   uint aq[4][8];vec2 ad[4];
   for(uint i=0;i<4;i++){uint a=(s*64+fx+16*i)*9;ad[i]=unpackHalf2x16(as_[a]);for(uint g=0;g<8;g++)aq[i][g]=as_[a+1+g];}
   for(uint j=0;j<8;j++){
    uint b=(s*64+ty+8*j)*9;vec2 q=unpackHalf2x16(bs_[b]);uint bq[8];for(uint g=0;g<8;g++)bq[g]=bs_[b+1+g];
    for(uint i=0;i<4;i++){int sum=0;for(uint g=0;g<8;g++)sum+=dotPacked4x8EXT(aq[i][g],int(bq[g]));acc[i][j]=fma(ad[i].x,corrected(sum,q),acc[i][j]);}
   }
  }
  barrier();
 }
 for(uint j=0;j<8;j++){uint row=row0+ty+8*j;if(row<M){
  for(uint i=0;i<4;i++){uint c=col0+fx+16*i;if(c<N)y[row*N+c]=acc[i][j]+bias[c];}}}
}
