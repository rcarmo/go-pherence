#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) readonly buffer T {float xt[];};
layout(set=0,binding=2) readonly buffer W {float weight[];};
layout(set=0,binding=3) readonly buffer B {float bias[];};
layout(set=0,binding=4) readonly buffer S {float stats[];};
layout(set=0,binding=5) writeonly buffer O {float result[];};
// mode0=stats[M,5]; mode1=value[M,N];mode2=tangent[M,N]. kind0=LN,kind1=variance.
layout(push_constant) uniform Args {uint rows;uint width;uint kind;uint mode;float eps;} a;
// Correct the native divide estimate with its FMA residual.
float trainingDivide(float n,float d){
 precise float q=n/d;
 precise float residual=fma(-q,d,n);
 return fma(residual,1.0/d,q);
}
// Compensated Newton correction avoids separate sqrt/divide rounding.
float trainingInvSqrt(float v){
 precise float r=inversesqrt(v);
 precise float rr=r*r;
 precise float squareError=fma(r,r,-rr);
 precise float residual=fma(-v,rr,1.0);
 residual=fma(-v,squareError,residual);
 return fma(0.5*r,residual,r);
}
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==0u){
  if(i>=a.rows)return;uint base=i*a.width;
  precise float mean=0.0,dmean=0.0;
  for(uint c=0u;c<a.width;c++){mean=mean+x[base+c];dmean=dmean+xt[base+c];}
  mean=trainingDivide(mean,float(a.width));dmean=trainingDivide(dmean,float(a.width));
  precise float variance=0.0,dvariance=0.0;
  for(uint c=0u;c<a.width;c++){
   precise float center=x[base+c]-mean,dcenter=xt[base+c]-dmean;
   variance=variance+center*center;dvariance=dvariance+(2.0*center)*dcenter;
  }
  float denom=a.kind==0u?float(a.width):float(a.width-1u);
  variance=trainingDivide(variance,denom);dvariance=trainingDivide(dvariance,denom);
  precise float inv=trainingInvSqrt(variance+a.eps);
  precise float dinv=((((-0.5)*inv)*inv)*inv)*dvariance;
  result[i*5u]=mean;result[i*5u+1u]=dmean;result[i*5u+2u]=inv;result[i*5u+3u]=dinv;result[i*5u+4u]=dvariance;
 }else{
  if(i>=a.rows*a.width)return;uint row=i/a.width,col=i%a.width;
  precise float value=a.kind==0u?x[i]-stats[row*5u]:x[i];
  precise float tangent=a.kind==0u?xt[i]-stats[row*5u+1u]:xt[i];
  if(a.mode==1u){
   precise float y=(value*weight[col])*stats[row*5u+2u];
   if(a.kind==0u)y=(value*stats[row*5u+2u])*weight[col]+bias[col];
   result[i]=y;
  }else{
   precise float yt;
   if(a.kind==0u){yt=weight[col]*(tangent*stats[row*5u+2u]+value*stats[row*5u+3u]);}
   else{yt=weight[col]*(tangent*stats[row*5u+2u]+value*stats[row*5u+3u]);}
   result[i]=yt;
  }
 }
}
