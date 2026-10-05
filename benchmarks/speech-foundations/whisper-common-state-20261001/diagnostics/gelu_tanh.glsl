#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) buffer O { float y[]; };
layout(push_constant) uniform P { uint count; };
// Diagnostic original ggml Vulkan GELU expression, baseline admission only.
void main(){uint i=gl_GlobalInvocationID.x;if(i>=count)return;float v=x[i];y[i]=0.5*v*(2.0-2.0/(1.0+exp(2.0*0.7978845608028654*v*(1.0+0.044715*v*v))));}
