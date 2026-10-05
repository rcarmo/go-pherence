#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer F {float freq[];};
layout(set=0,binding=1) readonly buffer T {float times[];};
layout(set=0,binding=2) readonly buffer A {float av[];};
layout(set=0,binding=3) readonly buffer B {float bv[];};
layout(set=0,binding=4) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint halfSize;uint mode;} a;
// Explicit F32 arithmetic for the admitted finite |angle| <= 8192 domain.
// Reduce with a split pi/2 and FMA; evaluate Taylor polynomials only on
// [-pi/4, pi/4]. Remainders are bounded by (pi/4)^15/15! and (pi/4)^16/16!;
// differential tests separately measure coefficient and F32 rounding.
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
void trainingSinCos(float angle,out float sine,out float cosine){
 precise float x=abs(angle),r=angle,rl=0.0;
 uint quadrant=0u;
 if(x>0.7853981633974483){
  quadrant=uint(fma(x,0.6366197723675814,0.5));
  precise float q=float(quadrant),p,pl;
  mulPair(q,0.0,1.570796251296997,7.549789415861596e-8,p,pl);
  addPair(x,0.0,-p,-pl,r,rl);
  addPair(r,rl,-q*5.390302529957765e-15,0.0,r,rl);
 }
 precise float z,zl;mulPair(r,rl,r,rl,z,zl);
 precise float st=r,stl=rl,ct=1.0,ctl=0.0,sh=r,sl=rl,ch=1.0,cl=0.0;
 for(uint n=1u;n<10u;n++){
  mulPair(st,stl,-z,-zl,st,stl);
  precise float rh=1.0,rlow=0.0;
  if(n==1u){rh=0.1666666716337204;rlow=-4.967053879312289e-9;}
  if(n==2u){rh=0.05000000074505806;rlow=-7.450580707946131e-10;}
  if(n==3u){rh=0.02380952425301075;rlow=-4.434869416147791e-10;}
  if(n==4u){rh=0.013888888992369175;rlow=-1.0348028683937471e-10;}
  if(n==5u){rh=0.00909090880304575;rlow=2.8786334427266524e-10;}
  if(n==6u){rh=0.006410256493836641;rlow=-8.358023301235917e-11;}
  if(n==7u){rh=0.004761904943734407;rlow=-1.8182964856006123e-10;}
  if(n==8u){rh=0.0036764706019312143;rlow=-1.3695920265954786e-11;}
  if(n==9u){rh=0.002923976629972458;rlow=-2.178532354513152e-11;}
  mulPair(st,stl,rh,rlow,st,stl);
  addPair(sh,sl,st,stl,sh,sl);
  mulPair(ct,ctl,-z,-zl,ct,ctl);
  rh=1.0;rlow=0.0;
  if(n==1u){rh=0.5;rlow=0;}
  if(n==2u){rh=0.0833333358168602;rlow=-2.4835269396561444e-9;}
  if(n==3u){rh=0.03333333507180214;rlow=-1.73846881335038e-9;}
  if(n==4u){rh=0.01785714365541935;rlow=-7.982765004577175e-10;}
  if(n==5u){rh=0.011111111380159855;rlow=-2.690487554968257e-10;}
  if(n==6u){rh=0.007575757801532745;rlow=-2.2577517633237676e-10;}
  if(n==7u){rh=0.005494505632668734;rlow=-1.3816324473392427e-10;}
  if(n==8u){rh=0.004166666883975267;rlow=-2.173086016687975e-10;}
  if(n==9u){rh=0.003267973894253373;rlow=-3.804422402109431e-11;}
  mulPair(ct,ctl,rh,rlow,ct,ctl);
  addPair(ch,cl,ct,ctl,ch,cl);
 }
 precise float sv=sh+sl,cv=ch+cl;
 uint k=quadrant&3u;
 if(k==0u){sine=sv;cosine=cv;}else if(k==1u){sine=cv;cosine=-sv;}
 else if(k==2u){sine=-sv;cosine=-cv;}else{sine=-cv;cosine=sv;}
 if(x>0.7853981633974483){if(angle<0.0)sine=-sine;}
}
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode<2u){
  if(i>=2u*a.halfSize)return;
  uint j=i%a.halfSize;precise float f=freq[j];precise float angle=times[0]*f;
  precise float s,c;trainingSinCos(angle,s,c);
  if(a.mode==0u){if(i<a.halfSize){result[i]=c;}else{result[i]=s;}}
  else {precise float derivative;if(i<a.halfSize){derivative=-s*f;}else{derivative=c*f;}result[i]=derivative*times[1];}
 }else{
  if(i!=0u)return;
  precise float sum=0.0;
  for(uint j=0u;j<a.halfSize;j++){
   precise float f=freq[j],angle=times[0]*f;
   precise float s,c;trainingSinCos(angle,s,c);
   precise float fc=-s*f,fs=c*f,sc=-c*f*f,ss=-s*f*f;
   sum+=av[j]*fc+av[a.halfSize+j]*fs;
   sum+=bv[j]*sc*times[1]+bv[a.halfSize+j]*ss*times[1];
  }
  result[0]=sum;
 }
}
