#version 450
// Analytic reverse-over-forward population LayerNorm or sample-variance norm.
// stats per row: mean,dmean,inv,dinv,dvariance. Forward supplies these values.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) readonly buffer T {float xt[];};
layout(set=0,binding=2) readonly buffer W {float weight[];};
layout(set=0,binding=3) readonly buffer G {float gv[];};
layout(set=0,binding=4) readonly buffer H {float gt[];};
layout(set=0,binding=5) readonly buffer S {float stats[];};
layout(set=0,binding=6) writeonly buffer O {float result[];};
// kind0=population LayerNorm with affine; kind1=uncentred x*alpha*inv with
// centred unbiased(sample) variance. mode0=dXvalue,1=dXtangent,2=dWeight,3=dBias.
layout(push_constant) uniform Args {uint rows;uint width;uint kind;uint mode;} a;
float trainingDivide(float n,float d){
 precise float q=n/d;
 precise float residual=fma(-q,d,n);
 return fma(residual,1.0/d,q);
}
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode>=2u){
  if(i>=a.width)return;precise float sum=0.0;
  for(uint r=0u;r<a.rows;r++){
   uint base=r*a.width;float mean=stats[r*5u],dmean=stats[r*5u+1u],inv=stats[r*5u+2u],dinv=stats[r*5u+3u];
   if(a.mode==3u)sum=sum+gv[base+i];
   else{
    precise float value=a.kind==0u?x[base+i]-mean:x[base+i];
    precise float tangent=a.kind==0u?xt[base+i]-dmean:xt[base+i];
    precise float tb=tangent*inv+value*dinv;
    precise float baseValue=value*inv;
    if(a.kind==0u){sum=sum+gv[base+i]*baseValue+gt[base+i]*tb;}
    else{sum=sum+(gv[base+i]*value)*inv+gt[base+i]*tb;}
   }
  }
  result[i]=sum;return;
 }
 // One invocation owns a complete row: no shared storage, no float atomics.
 if(i>=a.rows)return;uint base=i*a.width;
 float mean=stats[i*5u],dmean=stats[i*5u+1u],inv=stats[i*5u+2u],dinv=stats[i*5u+3u],dvar=stats[i*5u+4u];
 // For a primal-only LayerNorm VJP, use the centred normalized-base
 // expression directly. This matches the CPU association and avoids a
 // cancellation-prone reverse-over-forward expansion with zero tangents.
 {
  uint hasTangent=0u;
  precise float sumG=0.0,dotG=0.0;
  for(uint c=0u;c<a.width;c++){
   if(xt[base+c]==0.0){}else{hasTangent=1u;}
   if(gt[base+c]==0.0){}else{hasTangent=1u;}
   precise float normalized=(x[base+c]-mean)*inv;
   precise float g=gv[base+c]*weight[c];
   sumG=sumG+g;dotG=dotG+g*normalized;
  }
  if(hasTangent==0u){
   if(a.kind==0u){
    sumG=trainingDivide(sumG,float(a.width));dotG=trainingDivide(dotG,float(a.width));
    for(uint c=0u;c<a.width;c++){
     precise float normalized=(x[base+c]-mean)*inv;
     precise float g=gv[base+c]*weight[c];
     if(a.mode==0u){precise float derivative=inv*(g-sumG-normalized*dotG);result[base+c]=derivative;}
     else{result[base+c]=0.0;}
    }
   }else{
    precise float dscale=0.0;
    for(uint c=0u;c<a.width;c++)dscale=dscale+(gv[base+c]*x[base+c])*weight[c];
    precise float dvariance=(((dscale*(-0.5))*inv)*inv)*inv;
    for(uint c=0u;c<a.width;c++){
     precise float direct=(gv[base+c]*weight[c])*inv;
     precise float centered=(((dvariance*2.0)*(x[base+c]-mean))/float(a.width-1u));
     if(a.mode==0u){result[base+c]=direct+centered;}else{result[base+c]=0.0;}
    }
   }
   return;
  }
 }
 precise float ginv=0.0;precise float gdinv=0.0;
 for(uint c=0u;c<a.width;c++){
  precise float value=a.kind==0u?x[base+c]-mean:x[base+c];
  precise float tangent=a.kind==0u?xt[base+c]-dmean:xt[base+c];
  precise float g=gv[base+c]*weight[c],h=gt[base+c]*weight[c];
  ginv=ginv+(g*value+h*tangent);gdinv=gdinv+h*value;
 }
 precise float gvar=(((ginv*(-0.5))*inv)*inv)*inv;
 gvar=gvar+((((((gdinv*0.75)*inv)*inv)*inv)*inv)*inv)*dvar;
 precise float gdvar=(((gdinv*(-0.5))*inv)*inv)*inv;
 float denom=a.kind==0u?float(a.width):float(a.width-1u);
 precise float total=0.0;
 // LayerNorm input mean adjoints apply to all contributions. Variance norm
 // direct x*alpha*inv is uncentred: never subtract its mean derivative.
 if(a.kind==0u){
  for(uint c=0u;c<a.width;c++){
   precise float center=x[base+c]-mean,dcenter=xt[base+c]-dmean;
   precise float g=gv[base+c]*weight[c],h=gt[base+c]*weight[c];
   precise float value=a.mode==0u?(g*inv+h*dinv):h*inv;
   if(a.mode==0u)value=value+(((gvar*2.0)*center)/denom+((gdvar*2.0)*dcenter)/denom);
   else value=value+((gdvar*2.0)*center)/denom;
   total=total-value;
  }
 }
 for(uint c=0u;c<a.width;c++){
  precise float center=x[base+c]-mean,dcenter=xt[base+c]-dmean;
  precise float g=gv[base+c]*weight[c],h=gt[base+c]*weight[c];
  precise float value=a.mode==0u?(g*inv+h*dinv):h*inv;
  if(a.mode==0u){
   value=value+(((gvar*2.0)*center)/denom+((gdvar*2.0)*dcenter)/denom);
  }else value=value+((gdvar*2.0)*center)/denom;
  if(a.kind==0u)value=value+total/float(a.width);
  result[base+c]=value;
 }
}
