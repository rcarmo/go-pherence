#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer A {float previous[];};
layout(set=0,binding=1) readonly buffer B {float stats[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint n;uint mode;float previousCount;float maxNorm;} a;
bool finiteValue(float x){return abs(x)<=3.402823466e38;}
void main(){
 if(gl_GlobalInvocationID.x!=0u)return;
 float maximum=previous[0],sum=previous[1],invalid=previous[2],correction=previous[3];
 if(!finiteValue(maximum))invalid=1.0;
 if(!finiteValue(sum))invalid=1.0;
 if(!finiteValue(invalid))invalid=1.0;
 if(!finiteValue(correction))invalid=1.0;
 if(!(invalid==0.0))invalid=1.0;
 if(maximum<0.0)invalid=1.0;
 if(sum<0.0)invalid=1.0;
 if(maximum==0.0){if(!(sum==0.0))invalid=1.0;}
 if(sum>a.previousCount*(1.0+0.000002))invalid=1.0;
 if(a.mode==1u){
  if(maximum>0.0){if(sum<1.0)invalid=1.0;}
  float scale=1.0;
  if(invalid==0.0){if(maximum>0.0){float root=sqrt(sum),ratio=a.maxNorm/maximum;if(ratio<root)scale=ratio/root;}}
  if(!(invalid==0.0))scale=0.0;
  result[0]=maximum;result[1]=sum;result[2]=invalid;result[3]=scale;return;
 }
 uint chunks=(a.n-1u)/256u+1u;
 for(uint j=0u;j<chunks;j++){
  float m=stats[3u*j],s=stats[3u*j+1u],flag=stats[3u*j+2u];uint count=a.n-j*256u;if(count>256u)count=256u;
  if(!finiteValue(m))invalid=1.0;
  if(!finiteValue(s))invalid=1.0;
  if(!finiteValue(flag))invalid=1.0;
  if(!(flag==0.0))invalid=1.0;
  if(m<0.0)invalid=1.0;
  if(s<0.0)invalid=1.0;
  if(s>float(count))invalid=1.0;
  if(m==0.0){if(!(s==0.0))invalid=1.0;}else{if(s<1.0)invalid=1.0;}
  if(a.mode==0u){if(m>maximum)maximum=m;}else{if(m>maximum)invalid=1.0;}
 }
 precise float total=sum;
 precise float comp=correction;
 if(a.mode==0u){total=0.0;comp=0.0;}
 else{if(invalid==0.0){if(maximum>0.0){
  for(uint j=0u;j<chunks;j++){// Keep the chunk holding the global maximum at the exact unit ratio.
   precise float r=stats[3u*j]==maximum?1.0:stats[3u*j]/maximum;precise float term=(r*r)*stats[3u*j+1u]-comp;precise float next=total+term;comp=(next-total)-term;total=next;}
 }}}
 result[0]=maximum;result[1]=total;result[2]=invalid;result[3]=comp;
}
