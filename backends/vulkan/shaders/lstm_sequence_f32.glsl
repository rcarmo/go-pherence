#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Input { float input_data[]; };
layout(set=0,binding=1) readonly buffer WeightIH { float weight_ih[]; };
layout(set=0,binding=2) readonly buffer WeightHH { float weight_hh[]; };
layout(set=0,binding=3) readonly buffer BiasIH { float bias_ih[]; };
layout(set=0,binding=4) readonly buffer BiasHH { float bias_hh[]; };
layout(set=0,binding=5) buffer HiddenState { float hidden_state[]; };
layout(set=0,binding=6) buffer CellState { float cell_state[]; };
layout(set=0,binding=7) buffer Output { float output_data[]; };
layout(push_constant) uniform P {
    uint frames;
    uint inputDim;
    uint hidden;
    uint outputWidth;
    uint outputOffset;
    uint reverse;
    uint packedWeights;
};

// Complete one-direction PyTorch-IFGO recurrence. One workgroup owns a sequence;
// hidden<=256. Distinct input/hidden projections and biases preserve composition.
// Packed weights retain the declared [rows,columns] shape but store columns first,
// so adjacent lanes read adjacent rows while each lane preserves its column order.
shared float previous_hidden[256];
float stable_sigmoid(float value) {
    if (value >= 0.0) return 1.0 / (1.0 + exp(-value));
    float e = exp(value);
    return e / (1.0 + e);
}
float stable_tanh(float value) {
    if (value >= 0.0) {
        float e = exp(-2.0 * value);
        return (1.0 - e) / (1.0 + e);
    }
    float e = exp(2.0 * value);
    return (e - 1.0) / (e + 1.0);
}
void main() {
    uint unit=gl_LocalInvocationID.x;
    float cell=0.0;
    if (unit<hidden) {
        previous_hidden[unit]=hidden_state[unit];
        cell=cell_state[unit];
    }
    barrier();
    for (uint step=0;step<frames;step++) {
        precise float ii=0.0,fi=0.0,gi=0.0,oi=0.0;
        precise float ih=0.0,fh=0.0,gh=0.0,oh=0.0;
        uint frame=step;
        if (reverse!=0) frame=frames-1-step;
        if (unit<hidden) {
            uint r0=unit;
            uint r1=hidden+unit;
            uint r2=2*hidden+unit;
            uint r3=3*hidden+unit;
            uint inputBase=frame*inputDim;
            for (uint k=0;k<inputDim;k++) {
                float v=input_data[inputBase+k];
                uint b=packedWeights!=0 ? k*(4*hidden) : k;
                uint s=packedWeights!=0 ? 1 : inputDim;
                ii+=v*weight_ih[b+r0*s];
                fi+=v*weight_ih[b+r1*s];
                gi+=v*weight_ih[b+r2*s];
                oi+=v*weight_ih[b+r3*s];
            }
            for (uint k=0;k<hidden;k++) {
                float v=previous_hidden[k];
                uint b=packedWeights!=0 ? k*(4*hidden) : k;
                uint s=packedWeights!=0 ? 1 : hidden;
                ih+=v*weight_hh[b+r0*s];
                fh+=v*weight_hh[b+r1*s];
                gh+=v*weight_hh[b+r2*s];
                oh+=v*weight_hh[b+r3*s];
            }
            ii=(ii+bias_ih[r0])+(ih+bias_hh[r0]);
            fi=(fi+bias_ih[r1])+(fh+bias_hh[r1]);
            gi=(gi+bias_ih[r2])+(gh+bias_hh[r2]);
            oi=(oi+bias_ih[r3])+(oh+bias_hh[r3]);
        }
        barrier();
        if (unit<hidden) {
            float inputGate=stable_sigmoid(ii);
            float forgetGate=stable_sigmoid(fi);
            float candidate=stable_tanh(gi);
            float outputGate=stable_sigmoid(oi);
            cell=forgetGate*cell + inputGate*candidate;
            float nextHidden=outputGate*stable_tanh(cell);
            previous_hidden[unit]=nextHidden;
            output_data[frame*outputWidth+outputOffset+unit]=nextHidden;
        }
        barrier();
    }
    if (unit<hidden) {
        hidden_state[unit]=previous_hidden[unit];
        cell_state[unit]=cell;
    }
}
