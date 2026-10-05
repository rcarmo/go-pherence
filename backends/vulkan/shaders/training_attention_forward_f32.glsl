#version 450
// Exact causal/context keys, serial stable softmax; one writer per output.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Q {float qv[];};
layout(set=0,binding=1) readonly buffer K {float kv[];};
layout(set=0,binding=2) readonly buffer V {float vv[];};
layout(set=0,binding=3) readonly buffer A {float av[];};
layout(set=0,binding=4) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint heads;uint dim;uint contextRows;uint mode;float scale;} a;
bool keyAllowed(uint query,uint key){return key<query+1u&&(a.contextRows==0u||query-key<a.contextRows);}
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
float trainingExp(float v){
 if(v<-103.972084){return 0.0;}
 precise float hi,lo;expNegative(v,0.0,hi,lo);return hi+lo;
}
float trainingDivide(float n,float d){
 precise float q=n/d;
 precise float residual=fma(-q,d,n);
 return fma(residual,1.0/d,q);
}
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==0u){
  if(i>=a.heads*a.rows*a.rows)return;uint key=i%a.rows;uint query=(i/a.rows)%a.rows;uint head=i/(a.rows*a.rows);
  if(!keyAllowed(query,key)){result[i]=0.0;return;}
  precise float sum=0.0;for(uint d=0u;d<a.dim;d++)sum+=qv[(query*a.heads+head)*a.dim+d]*kv[(key*a.heads+head)*a.dim+d];result[i]=sum*a.scale;
 }else if(a.mode==1u){
  if(i>=a.heads*a.rows)return;uint query=i%a.rows;uint base=i*a.rows;
  float maximum=-3.402823466e38;for(uint key=0u;key<a.rows;key++){if(keyAllowed(query,key)&&av[base+key]>maximum)maximum=av[base+key];}
  precise float sum=0.0;for(uint key=0u;key<a.rows;key++){if(keyAllowed(query,key))sum+=trainingExp(av[base+key]-maximum);}
  for(uint key=0u;key<a.rows;key++)result[base+key]=keyAllowed(query,key)?trainingDivide(trainingExp(av[base+key]-maximum),sum):0.0;
 }else{
  if(i>=a.rows*a.heads*a.dim)return;uint d=i%a.dim;uint head=(i/a.dim)%a.heads;uint query=i/(a.heads*a.dim);
  precise float sum=0.0;for(uint key=0u;key<a.rows;key++){if(keyAllowed(query,key))sum+=av[(head*a.rows+query)*a.rows+key]*vv[(key*a.heads+head)*a.dim+d];}result[i]=sum;
 }
}
