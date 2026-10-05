#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) buffer P { float parameter[]; };
layout(set=0,binding=1) readonly buffer G { float gradient[]; };
layout(set=0,binding=2) buffer M { float moment[]; };
layout(set=0,binding=3) buffer V { float variance[]; };
layout(set=0,binding=4) buffer E { float ema[]; };
layout(set=0,binding=5) readonly buffer C { float clip[]; };
layout(push_constant) uniform Args {
 uint count; float lr; float beta1; float beta2; float eps; float decay;
 float bias1; float bias2; float emaDecay; float reservedScale;
} a;
void main() {
 uint i=gl_GlobalInvocationID.x;
 if(i>=a.count) return;
 precise float g=gradient[i]*clip[3];
 precise float m=a.beta1*moment[i]+(1.0-a.beta1)*g;
 precise float v=a.beta2*variance[i]+((1.0-a.beta2)*g)*g;
 precise float mh=m/a.bias1;
 precise float vh=v/a.bias2;
 precise float p=parameter[i]*(1.0-a.lr*a.decay);
 p=p-(a.lr*mh)/(sqrt(vh)+a.eps);
 moment[i]=m; variance[i]=v; parameter[i]=p;
 if(a.emaDecay>0.0) {
  precise float shadow=a.emaDecay*ema[i]+(1.0-a.emaDecay)*p;
  ema[i]=shadow;
 }
}
