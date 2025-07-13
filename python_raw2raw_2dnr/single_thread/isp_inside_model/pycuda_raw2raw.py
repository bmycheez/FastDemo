import pycuda.autoinit
import pycuda.driver as cuda
import subprocess
import time
import numpy as np
import tensorrt as trt
import cv2

# config
read_shape = (1112, 2028)
outp_shape = (3, 544, 960)
frame_rate = 60
subprocess.call(['v4l2-ctl', '-c', 'sensor_mode=6'])
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"BG10"))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, read_shape[1])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, read_shape[0])
cam.set(cv2.CAP_PROP_FPS, frame_rate)
bool_sync = True
total_frames = 100

# load engine
ctx = cuda.Device(0).make_context()
stream = cuda.Stream()
trt_logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(trt_logger)
with open('Unet4to4-8ch_alpha-1.0.engine', "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
size = []
dtype = []
for binding in engine:
    size.append(trt.volume(engine.get_binding_shape(binding)))
    dtype.append(trt.nptype(engine.get_binding_dtype(binding)))
print(dtype)

# allocate buffers
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    
    host_mem = cuda.pagelocked_empty(size, dtype)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)
    
    bindings.append(int(cuda_mem))
    
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)

i = total_frames

# FPS
runtime1 = []
runtime2 = []
runtime3 = []
runtime4 = []
runtime5 = []
runtime6 = []
runtime7 = []
runtime8 = []
runtimeT = []


while i != 0:
    start0 = time.time()
    
    # read
    _, noisy_frame = cam.read()
    noisy_frame = noisy_frame.ravel()
    noisy_frame = noisy_frame.view(np.uint16)[:read_shape[0]*read_shape[1]]
    noisy_frame = noisy_frame >> 6
    
    end = time.time()
    runtime = end - start0
    runtime1.append(runtime)
    
    start = time.time()
    
    # HWC > CHW (이미 벡터이므로 필요 없음)
    
    
    end = time.time()
    runtime = end - start
    runtime2.append(runtime)
    
    start = time.time()
    
    # change dtype
    noisy_frame = noisy_frame.astype(np.float32)
    
    end = time.time()
    runtime = end - start
    runtime3.append(runtime)
    
    start = time.time()
    
    # H2D copy
    noisy_frame = np.ascontiguousarray(noisy_frame)
    np.copyto(host_inputs[0], noisy_frame)
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    if bool_sync:
        stream.synchronize()
    
    end = time.time()
    runtime = end - start
    runtime4.append(runtime)
    
    start = time.time()
    
    # model
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    if bool_sync:
        stream.synchronize()
    
    end = time.time()
    runtime = end - start
    runtime5.append(runtime)
    
    start = time.time()
    
    # d2h
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    denoised_frame = host_outputs[0].reshape(outp_shape)
    if bool_sync:
        stream.synchronize()
        
    end = time.time()
    runtime = end - start
    runtime6.append(runtime)
    
    start = time.time()
    
    # CHW > HWC
    denoised_frame = np.moveaxis(denoised_frame, 0, 2)
    
    end = time.time()
    runtime = end - start
    runtime7.append(runtime)
    
    start = time.time()
    
    # display
    denoised_frame = cv2.resize(denoised_frame, dsize=(outp_shape[2]*2, outp_shape[1]*2))
    cv2.imshow('PyCUDA', denoised_frame)
    keyCode = cv2.waitKey(1) & 0xFF
    if keyCode == 27 or keyCode == ord('q'):
        break
    i -= 1
    
    end = time.time()
    runtime = end - start
    runtime8.append(runtime)
    
    end0 = time.time()
    runtime = end0 - start0
    runtimeT.append(runtime)
print(f'Read: {sum(runtime1)/len(runtime1)*1000:.4f}ms')
print(f'HWC > CHW: {sum(runtime2)/len(runtime2)*1000:.4f}ms')
print(f'change dtype: {sum(runtime3)/len(runtime3)*1000:.4f}ms')
print(f'H2D copy: {sum(runtime4)/len(runtime4)*1000:.4f}ms')
print(f'Model: {sum(runtime5)/len(runtime5)*1000:.4f}ms')
print(f'D2H copy: {sum(runtime6)/len(runtime6)*1000:.4f}ms')
print(f'CHW > HWC: {sum(runtime7)/len(runtime7)*1000:.4f}ms')
print(f'Display: {sum(runtime8)/len(runtime8)*1000:.4f}ms')
print(f'Total (FPS): {sum(runtimeT)/len(runtimeT)*1000:.4f}ms / {len(runtimeT)/sum(runtimeT):.4f}')
cam.release()
cv2.destroyAllWindows()
