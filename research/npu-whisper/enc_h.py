import numpy as np, onnxruntime as ort, sys, time
mel=np.fromfile(sys.argv[1],dtype=np.float32).reshape(1,128,3000)
so=ort.SessionOptions();so.intra_op_num_threads=6
e=ort.InferenceSession('enc_turbo.onnx',so,providers=['CPUExecutionProvider'])
t=time.time();H=e.run(['last_hidden_state'],{'input_features':mel})[0]
print('turbo encoder %.1fs'%(time.time()-t),H.shape)
np.save('H_turbo.npy',H)
