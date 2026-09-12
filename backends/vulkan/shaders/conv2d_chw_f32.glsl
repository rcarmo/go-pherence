#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float weight[]; };
layout(set=0,binding=2) buffer O { float output_data[]; };
layout(push_constant) uniform P {
    uint inChannels;
    uint inFrequency;
    uint inFrames;
    uint outChannels;
    uint outFrequency;
    uint outFrames;
    uint kernel;
    uint stride;
    uint padding;
};

// Bias-free CHW convolution for kernel1/pad0 or kernel3/pad1, stride1/2.
// One workgroup computes a 16-position by 16-output-channel tile. The shared
// implicit-im2col tiles avoid host transposition or a materialised patch tensor.
shared float xt[256];
shared float wt[256];
void main() {
    uint cx=gl_LocalInvocationID.x, ry=gl_LocalInvocationID.y;
    uint position=gl_WorkGroupID.y*16+ry;
    uint outputChannel=gl_WorkGroupID.x*16+cx;
    uint outSpatial=outFrequency*outFrames;
    uint reduction=inChannels*kernel*kernel;
    precise float sum=0.0;
    for (uint base=0; base<reduction; base+=16) {
        uint reductionIndex=base+cx;
        float value=0.0;
        if (position<outSpatial) {
            if (reductionIndex<reduction) {
                uint outF=position/outFrames;
                uint outT=position%outFrames;
                uint tapArea=kernel*kernel;
                uint inputChannel=reductionIndex/tapArea;
                uint tap=reductionIndex%tapArea;
                uint tapF=tap/kernel;
                uint tapT=tap%kernel;
                uint paddedF=outF*stride+tapF;
                uint paddedT=outT*stride+tapT;
                if (paddedF>=padding) {
                    if (paddedT>=padding) {
                        uint inF=paddedF-padding;
                        uint inT=paddedT-padding;
                        if (inF<inFrequency) {
                            if (inT<inFrames)
                                value=x[(inputChannel*inFrequency+inF)*inFrames+inT];
                        }
                    }
                }
            }
        }
        xt[ry*16+cx]=value;
        uint weightChannel=gl_WorkGroupID.x*16+ry;
        value=0.0;
        if (weightChannel<outChannels) {
            if (reductionIndex<reduction)
                value=weight[weightChannel*reduction+reductionIndex];
        }
        wt[ry*16+cx]=value;
        barrier();
        for (uint j=0;j<16;j++) sum+=xt[ry*16+j]*wt[cx*16+j];
        barrier();
    }
    if (position<outSpatial) {
        if (outputChannel<outChannels)
            output_data[outputChannel*outSpatial+position]=sum;
    }
}
