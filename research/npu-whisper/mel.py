import numpy as np, json, sys

# --- Whisper log-mel (128 bins), numpy port matching openai/transformers ---
def hz_to_mel_slaney(freq):
    f_min, f_sp = 0.0, 200.0/3
    mels = (freq - f_min)/f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0-f_min)/f_sp
    logstep = np.log(6.4)/27.0
    mels = np.where(freq >= min_log_hz, min_log_mel + np.log(freq/min_log_hz)/logstep, mels)
    return mels

def mel_to_hz_slaney(mels):
    f_min, f_sp = 0.0, 200.0/3
    freqs = f_min + f_sp*mels
    min_log_hz, min_log_mel = 1000.0, (1000.0-f_min)/f_sp
    logstep = np.log(6.4)/27.0
    freqs = np.where(mels >= min_log_mel, min_log_hz*np.exp(logstep*(mels-min_log_mel)), freqs)
    return freqs

def mel_filterbank(sr=16000, n_fft=400, n_mels=128):
    fmax = sr/2.0
    n_freqs = n_fft//2 + 1
    fftfreqs = np.linspace(0, sr/2.0, n_freqs)
    mel_min, mel_max = hz_to_mel_slaney(np.array([0.0])), hz_to_mel_slaney(np.array([fmax]))
    mel_pts = np.linspace(mel_min[0], mel_max[0], n_mels+2)
    freq_pts = mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(freq_pts)
    ramps = freq_pts.reshape(-1,1) - fftfreqs.reshape(1,-1)
    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i]/fdiff[i]
        upper = ramps[i+2]/fdiff[i+1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # slaney norm
    enorm = 2.0/(freq_pts[2:n_mels+2]-freq_pts[:n_mels])
    weights *= enorm.reshape(-1,1)
    return weights.astype(np.float32)

def log_mel(audio, n_mels=128):
    n_fft, hop = 400, 160
    audio = audio.astype(np.float32)
    window = np.hanning(n_fft+1)[:-1].astype(np.float32)  # periodic hann
    pad = n_fft//2
    a = np.pad(audio, (pad,pad), mode='reflect')
    n_frames = 1 + (len(a)-n_fft)//hop
    frames = np.stack([a[i*hop:i*hop+n_fft]*window for i in range(n_frames)], axis=1)  # [n_fft, frames]
    stft = np.fft.rfft(frames, axis=0)  # [201, frames]
    mag = (np.abs(stft)**2)[:, :-1]  # drop last frame -> [201, 3000] for 30s
    fb = mel_filterbank(16000, n_fft, n_mels)
    mel = fb @ mag
    logs = np.log10(np.maximum(mel, 1e-10))
    logs = np.maximum(logs, logs.max()-8.0)
    logs = (logs+4.0)/4.0
    return logs.astype(np.float32)  # [n_mels, frames]

def load_wav(path):
    import wave
    w=wave.open(path,'rb'); sr=w.getframerate(); n=w.getnframes()
    raw=w.readframes(n); w.close()
    a=np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
    if w.getnchannels()==2: a=a.reshape(-1,2).mean(1)
    assert sr==16000, sr
    return a

def mel30(path):
    a=load_wav(path)
    target=480000
    if len(a)<target: a=np.pad(a,(0,target-len(a)))
    else: a=a[:target]
    return log_mel(a)[None,:,:]  # [1,128,3000]

if __name__=='__main__':
    m=mel30(sys.argv[1]); print('mel', m.shape, m.dtype, 'range', float(m.min()), float(m.max()))
    m.tofile(sys.argv[2] if len(sys.argv)>2 else 'mel.bin')
