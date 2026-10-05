#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) readonly buffer L {float labels[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint n;uint activeCount;uint mode;float weight;} a;
// log(1+e) series avoids losing small softplus tails when 1+e rounds to1.
float log1pPositive(float e){
 if(e>=0.125)return log(1.0+e);
 precise float term=e,sum=e;
 for(uint k=2u;k<13u;k++){term*=e;float value=term/float(k);if((k%2u)==0u){sum-=value;}else{sum+=value;}}
 return sum;
}
void addPair(float a,float al,float b,float bl,precise out float s,precise out float sl){
 precise float z=a+b,v=z-a;
 precise float low=((a-(z-v))+(b-v))+(al+bl);
 precise float t=z+low;
 precise float tail=low-(t-z);
 s=t;sl=tail;
}
void mulPair(float a,float al,float b,float bl,precise out float p,precise out float pl){
 precise float z=a*b;
 precise float low=fma(a,b,-z)+(a*bl+al*b);
 precise float t=z+low;
 precise float tail=low-(t-z);
 p=t;pl=tail;
}
void divPair(float n,float nl,float d,float dl,precise out float q,precise out float ql){
 precise float z=n/d;
 precise float low=(fma(-z,d,n)+(nl-z*dl))/d;
 precise float t=z+low;
 precise float tail=low-(t-z);
 q=t;ql=tail;
}
void expNegative(float v,float vl,precise out float e,precise out float el){
 // v<=0; F32 reduction with hi/lo argument and split ln2.
 uint k=uint(-v*1.4426950408889634+0.5);
 precise float fk=float(k),r,rl,lh,ll;
 mulPair(fk,0.0,0.6931471824645996,-1.904654323148236e-9,lh,ll);
 addPair(v,vl,lh,ll,r,rl);
 precise float ph=1.0,pl=0.0,term=1.0,terml=0.0;
 // |r|<=ln2/2. Degree14 remainder <3e-20; retain low rounding terms.
 for(uint n=1u;n<15u;n++){
  mulPair(term,terml,r,rl,term,terml);
  precise float reciprocal=1.0,reciprocalLow=0.0;
  if(n==2u){reciprocal=0.5;reciprocalLow=0;}
  if(n==3u){reciprocal=0.3333333432674408;reciprocalLow=-9.934107758624577e-9;}
  if(n==4u){reciprocal=0.25;reciprocalLow=0;}
  if(n==5u){reciprocal=0.20000000298023224;reciprocalLow=-2.9802322831784522e-9;}
  if(n==6u){reciprocal=0.1666666716337204;reciprocalLow=-4.967053879312289e-9;}
  if(n==7u){reciprocal=0.1428571492433548;reciprocalLow=-6.38621200366174e-9;}
  if(n==8u){reciprocal=0.125;reciprocalLow=0;}
  if(n==9u){reciprocal=0.1111111119389534;reciprocalLow=-8.278422947149977e-10;}
  if(n==10u){reciprocal=0.10000000149011612;reciprocalLow=-1.4901161415892261e-9;}
  if(n==11u){reciprocal=0.09090909361839294;reciprocalLow=-2.709302115988521e-9;}
  if(n==12u){reciprocal=0.0833333358168602;reciprocalLow=-2.4835269396561444e-9;}
  if(n==13u){reciprocal=0.07692307978868484;reciprocalLow=-2.8656079731348427e-9;}
  if(n==14u){reciprocal=0.0714285746216774;reciprocalLow=-3.19310600183087e-9;}
  mulPair(term,terml,reciprocal,reciprocalLow,term,terml);
  addPair(ph,pl,term,terml,ph,pl);
 }
 precise float scale=0.0;
 if(k<127u){scale=uintBitsToFloat((127u-k)<<23u);}
 else if(k<150u){scale=uintBitsToFloat(1u<<(149u-k));}
 mulPair(ph,pl,scale,0.0,e,el);
}
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==2u){
  if(i!=0u)return;
  precise float sum=0.0,correction=0.0;
  for(uint j=0u;j<a.n;j++){if(labels[j]>=0.0){precise float term=xv[j]-correction;precise float next=sum+term;correction=(next-sum)-term;sum=next;}}
  result[0]=sum/float(a.activeCount);return;
 }
 if(i>=a.n)return;
 if(labels[i]<0.0){result[i]=0.0;return;}
 float x=xv[i];float e=exp(-abs(x));
 if(a.mode==0u){float base=0.0;if(labels[i]==1.0){if(x<0.0)base=-x;}else{if(x>0.0)base=x;}result[i]=base+log1pPositive(e);}
 else{
  precise float eh,el,dh,dl,sh,sl;
  expNegative(-abs(x),0.0,eh,el);
  addPair(1.0,0.0,eh,el,dh,dl);
  if(x<0.0){divPair(eh,el,dh,dl,sh,sl);}
  else{divPair(1.0,0.0,dh,dl,sh,sl);}
  addPair(sh,sl,-labels[i],0.0,sh,sl);
  // Match F64 CPU reciprocal-count multiplication, then F32 gradient rounding.
  precise float reciprocal=1.0/float(a.activeCount);
  precise float reciprocalLow=fma(-reciprocal,float(a.activeCount),1.0)/float(a.activeCount);
  mulPair(sh,sl,reciprocal,reciprocalLow,sh,sl);
  precise float gradient=sh+sl;result[i]=gradient*a.weight;
 }
}
