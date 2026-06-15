package ptx

// FFTPTX is a correctness-first fused mel spectrogram kernel.
// Grid: (num_frames, 1, 1), Block: at least num_mels threads (80/128 typical).
// Each thread computes one mel bin for one frame. Instead of an optimized shared
// radix-2 FFT, this body uses a direct DFT per frequency bin so the GPU graph is
// complete and faithful (window → spectrum → mel → log10) before later tiling/
// FFT optimization replaces the inner loops.
const FFTPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry mel_spectrogram(
    .param .u64 out_ptr,        // output: [num_mels * num_frames] float32
    .param .u64 audio_ptr,      // input: [total_samples] float32
    .param .u64 window_ptr,     // Hann window: [fft_size] float32
    .param .u64 mel_filters_ptr,// mel filterbank: [num_mels * num_bins] float32
    .param .u32 num_frames,
    .param .u32 fft_size,       // 400
    .param .u32 hop_length,     // 160
    .param .u32 num_mels,       // 80/128
    .param .u32 num_bins        // 257 (fft_padded/2 + 1)
) {
    .reg .pred %p_m_oob, %p_k_done, %p_n_done, %p_n_in_fft;
    .reg .u32 %frame, %m, %k, %n, %num_frames_u, %fft_size_u, %hop, %num_mels_u, %num_bins_u;
    .reg .u32 %audio_idx, %filter_idx, %out_idx;
    .reg .u64 %outp, %audiop, %winp, %filterp, %addr, %off;
    .reg .f32 %sample, %win, %x, %re, %im, %power, %mel, %filter;
    .reg .f32 %nf, %kf, %angle, %c, %s, %tmp, %log2v;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %audiop, [audio_ptr];
    ld.param.u64 %winp, [window_ptr];
    ld.param.u64 %filterp, [mel_filters_ptr];
    ld.param.u32 %num_frames_u, [num_frames];
    ld.param.u32 %fft_size_u, [fft_size];
    ld.param.u32 %hop, [hop_length];
    ld.param.u32 %num_mels_u, [num_mels];
    ld.param.u32 %num_bins_u, [num_bins];

    mov.u32 %frame, %ctaid.x;
    mov.u32 %m, %tid.x;
    setp.ge.u32 %p_m_oob, %m, %num_mels_u;
    @%p_m_oob bra DONE;

    mov.f32 %mel, 0f00000000;
    mov.u32 %k, 0;
K_LOOP:
    setp.ge.u32 %p_k_done, %k, %num_bins_u;
    @%p_k_done bra MEL_DONE;

    mov.f32 %re, 0f00000000;
    mov.f32 %im, 0f00000000;
    mov.u32 %n, 0;
N_LOOP:
    // Direct DFT over the logical 512-point FFT length. Samples beyond fft_size
    // are zero padding and therefore skipped.
    setp.ge.u32 %p_n_done, %n, 512;
    @%p_n_done bra DFT_DONE;
    setp.lt.u32 %p_n_in_fft, %n, %fft_size_u;
    @!%p_n_in_fft bra NEXT_N;

    mad.lo.u32 %audio_idx, %frame, %hop, %n;
    mul.wide.u32 %off, %audio_idx, 4;
    add.u64 %addr, %audiop, %off;
    ld.global.f32 %sample, [%addr];
    mul.wide.u32 %off, %n, 4;
    add.u64 %addr, %winp, %off;
    ld.global.f32 %win, [%addr];
    mul.rn.f32 %x, %sample, %win;

    cvt.rn.f32.u32 %nf, %n;
    cvt.rn.f32.u32 %kf, %k;
    mul.rn.f32 %angle, %kf, %nf;
    mul.rn.f32 %angle, %angle, 0fc0c90fdb; // -2*pi
    mul.rn.f32 %angle, %angle, 0f3b000000; // /512
    cos.approx.ftz.f32 %c, %angle;
    sin.approx.ftz.f32 %s, %angle;
    fma.rn.f32 %re, %x, %c, %re;
    fma.rn.f32 %im, %x, %s, %im;

NEXT_N:
    add.u32 %n, %n, 1;
    bra N_LOOP;
DFT_DONE:
    mul.rn.f32 %power, %re, %re;
    fma.rn.f32 %power, %im, %im, %power;
    mad.lo.u32 %filter_idx, %m, %num_bins_u, %k;
    mul.wide.u32 %off, %filter_idx, 4;
    add.u64 %addr, %filterp, %off;
    ld.global.f32 %filter, [%addr];
    fma.rn.f32 %mel, %filter, %power, %mel;
    add.u32 %k, %k, 1;
    bra K_LOOP;

MEL_DONE:
    // mel[m] = log10(max(mel[m], 1e-10)); log10(x)=lg2(x)*log10(2)
    max.f32 %mel, %mel, 0f2edbe6ff; // 1e-10f
    lg2.approx.ftz.f32 %log2v, %mel;
    mul.rn.f32 %mel, %log2v, 0f3e9a209b; // log10(2)
    mad.lo.u32 %out_idx, %m, %num_frames_u, %frame;
    mul.wide.u32 %off, %out_idx, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %mel;
DONE:
    ret;
}
`
