#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) readonly buffer G {float gv[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint count;uint mode;} a;
// F32 hi/lo operations retain coefficient and argument rounding through tanh.
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
 precise float scale=uintBitsToFloat((127u-k)<<23u);
 mulPair(ph,pl,scale,0.0,e,el);
}
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.count)return;
 precise float value=x[i];
 precise float z,zl,c,cl,u,ul;
 mulPair(value,0.0,value,0.0,z,zl);
 mulPair(z,zl,value,0.0,c,cl);
 // Split the exact decimal .044715 and sqrt(2/pi) coefficient used by CPU.
 if(value>8.0){if(a.mode==0u){result[i]=value;}else{result[i]=gv[i];}return;}
 if(value<-8.0){result[i]=0.0;return;}
 mulPair(c,cl,0.044715,1.5461444347053543e-9,u,ul);
 addPair(value,0.0,u,ul,u,ul);
 mulPair(u,ul,0.7978845834732056,-2.2670340271702116e-8,u,ul);
 precise float neg=-2.0*u,negl=-2.0*ul;
 if(u<0.0){neg=-neg;negl=-negl;}
 precise float e,el;expNegative(neg,negl,e,el);
 precise float numerator,numeratorl,denominator,denominatorl,th,tl;
 addPair(1.0,0.0,-e,-el,numerator,numeratorl);
 addPair(1.0,0.0,e,el,denominator,denominatorl);
 divPair(numerator,numeratorl,denominator,denominatorl,th,tl);
 if(value<0.0){th=-th;tl=-tl;}
 precise float onePlus,onePlusl,y,yl;
 addPair(1.0,0.0,th,tl,onePlus,onePlusl);
 if(a.mode==0u){
  mulPair(0.5*value,0.0,onePlus,onePlusl,y,yl);
 }else{
  precise float dt,dtl,du,dul,part,partl;
  mulPair(th,tl,th,tl,dt,dtl);
  addPair(1.0,0.0,-dt,-dtl,dt,dtl);
  mulPair(z,zl,0.10703222453594208,-1.2704169660704423e-10,du,dul);
  addPair(0.7978845834732056,-2.2670340271702116e-8,du,dul,du,dul);
  mulPair(dt,dtl,du,dul,part,partl);
  mulPair(0.5*value,0.0,part,partl,part,partl);
  mulPair(0.5,0.0,onePlus,onePlusl,y,yl);
  addPair(y,yl,part,partl,y,yl);
  // CPU rounds the derivative to F32 before multiplying the gradient seed.
  precise float derivative=y+yl;
  y=gv[i]*derivative;yl=0.0;
 }
 result[i]=y+yl;
}
