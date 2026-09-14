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
// One 16x16 workgroup computes a 32-position by 32-output-channel tile. Each
// invocation owns four outputs and shared 32-wide implicit-im2col tiles avoid
// host transposition or a materialised patch tensor.
shared float xt[1024];
shared float wt[1024];
void main() {
    uint cx=gl_LocalInvocationID.x, ry=gl_LocalInvocationID.y;
    uint lane=ry*16+cx;
    uint positionBase=gl_WorkGroupID.y*32;
    uint channelBase=gl_WorkGroupID.x*32;
    uint outSpatial=outFrequency*outFrames;
    uint reduction=inChannels*kernel*kernel;
    precise float s00=0.0, s01=0.0, s10=0.0, s11=0.0;
    for (uint base=0; base<reduction; base+=32) {
        for (uint i=lane;i<1024;i+=256) {
            uint tileRow=i/32;
            uint reductionIndex=base+i%32;
            float inputValue=0.0;
            uint position=positionBase+tileRow;
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
                                    inputValue=x[(inputChannel*inFrequency+inF)*inFrames+inT];
                            }
                        }
                    }
                }
            }
            xt[i]=inputValue;
            float weightValue=0.0;
            uint outputChannel=channelBase+tileRow;
            if (outputChannel<outChannels) {
                if (reductionIndex<reduction)
                    weightValue=weight[outputChannel*reduction+reductionIndex];
            }
            wt[i]=weightValue;
        }
        barrier();
        for (uint j=0;j<32;j++) {
            float a0=xt[ry*32+j], a1=xt[(ry+16)*32+j];
            float b0=wt[cx*32+j], b1=wt[(cx+16)*32+j];
            s00+=a0*b0;
            s01+=a0*b1;
            s10+=a1*b0;
            s11+=a1*b1;
        }
        barrier();
    }
    uint position=positionBase+ry;
    uint outputChannel=channelBase+cx;
    if (position<outSpatial) {
        if (outputChannel<outChannels) output_data[outputChannel*outSpatial+position]=s00;
        if (outputChannel+16<outChannels) output_data[(outputChannel+16)*outSpatial+position]=s01;
    }
    if (position+16<outSpatial) {
        if (outputChannel<outChannels) output_data[outputChannel*outSpatial+position+16]=s10;
        if (outputChannel+16<outChannels) output_data[(outputChannel+16)*outSpatial+position+16]=s11;
    }
}
