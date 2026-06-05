import numpy as np, sys, time, subprocess
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from mel import mel30
WAV='/home/me/go-pherence/testdata/podcast.wav'
mels=[]
for off in [30,150,300]:
    out=f'/home/me/npu/cal_{off}.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(off),'-t','30','-i',WAV,'-ar','16000','-ac','1',out],check=True)
    mels.append(mel30(out))
print('calib mels:',len(mels),flush=True)
class R(CalibrationDataReader):
    def __init__(s,m): s.it=iter([{'input_features':x} for x in m])
    def get_next(s): return next(s.it,None)
t=time.time()
quantize_static('enc_clean.onnx','enc_clean_static.onnx', R(mels),
    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True, calibrate_method=CalibrationMethod.MinMax, use_external_data_format=True,
    extra_options={'ActivationSymmetric':True,'WeightSymmetric':True})
print('static quantized in %.0fs'%(time.time()-t),flush=True)
