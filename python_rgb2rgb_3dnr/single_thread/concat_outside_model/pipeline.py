import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import time
import numpy as np
import tensorrt as trt
import cv2

# config
inp_shape = (21, 1088, 1920)
outp_shape = (3, 1088, 1920)
cam = cv2.VideoCapture(
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1088, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1920, height=(int)1088, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
    cv2.CAP_GSTREAMER)
bool_sync = True
total_frames = 100

# load engine
ctx = cuda.Device(0).make_context()
stream = cuda.Stream()
trt_logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(trt_logger)
with open('postprocessing_onnx-fp32_trt-fp16.engine', "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
size = []
dtype = []
for binding in engine:
    size.append(trt.volume(engine.get_binding_shape(binding)))
    dtype.append(trt.nptype(engine.get_binding_dtype(binding)))
print(dtype)

# input frames
inp_frames = np.empty(shape=inp_shape, dtype=dtype[0])
inp_arr = gpuarray.to_gpu(inp_frames)

# output frame
outp_frames = np.empty(shape=outp_shape, dtype=dtype[1])
outp_arr = gpuarray.to_gpu(outp_frames)

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
runtime9 = []
runtime0 = []
runtimeT = []

while i > 0:
    start0 = time.time()
    
    # read
    _, noisy_frame = cam.read()
    
    end = time.time()
    runtime = end - start0
    runtime1.append(runtime)
    
    start = time.time()
    
    # HWC > CHW
    noisy_frame = np.moveaxis(noisy_frame, 2, 0)
    
    end = time.time()
    runtime = end - start
    runtime2.append(runtime)
    
    start = time.time()
    
    # H2D copy
    noisy_frame = np.ascontiguousarray(noisy_frame)
    noisy_frame = gpuarray.to_gpu_async(ary=noisy_frame, stream=stream)
    if bool_sync:
        stream.synchronize()
    
    end = time.time()
    runtime = end - start
    runtime3.append(runtime)
    
    start = time.time()
    
    # change dtype
    noisy_frame = noisy_frame.astype(dtype[0])
    
    end = time.time()
    runtime = end - start
    runtime4.append(runtime)
    
    start = time.time()
    
    # concat
    inp_arr = gpuarray.concatenate((inp_arr[3:, :, :], noisy_frame), axis=0)
    
    end = time.time()
    runtime = end - start
    runtime5.append(runtime)
    
    start = time.time()
    
    # model
    bindings = [int(inp_arr.gpudata), int(outp_arr.gpudata)]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    if bool_sync:
        stream.synchronize()
    
    end = time.time()
    runtime = end - start
    runtime6.append(runtime)
    
    start = time.time()
    
    # change dtype
    outp_cpu = outp_arr.astype(np.uint8)
        
    end = time.time()
    runtime = end - start
    runtime7.append(runtime)
    
    start = time.time()
    
    # D2H copy
    outp_cpu = outp_cpu.get_async()
    
    end = time.time()
    runtime = end - start
    runtime8.append(runtime)
    
    start = time.time()
    
    # CHW > HWC
    outp_cpu = np.moveaxis(outp_cpu, 0, 2)
    
    end = time.time()
    runtime = end - start
    runtime9.append(runtime)
    
    start = time.time()
    
    # display
    cv2.imshow('PyCUDA', outp_cpu)
    keyCode = cv2.waitKey(1) & 0xFF
    if keyCode == 27 or keyCode == ord('q'):
        break
    i -= 1
    
    end = time.time()
    runtime = end - start
    runtime0.append(runtime)
    
    end0 = time.time()
    runtime = end0 - start0
    runtimeT.append(runtime)
print(f'Read: {sum(runtime1)/len(runtime1)*1000:.4f}ms')
print(f'HWC > CHW: {sum(runtime2)/len(runtime2)*1000:.4f}ms')
print(f'H2D copy: {sum(runtime3)/len(runtime3)*1000:.4f}ms')
print(f'change dtype: {sum(runtime4)/len(runtime4)*1000:.4f}ms')
print(f'concat: {sum(runtime5)/len(runtime5)*1000:.4f}ms')
print(f'Model: {sum(runtime6)/len(runtime6)*1000:.4f}ms')
print(f'change dtype: {sum(runtime7)/len(runtime7)*1000:.4f}ms')
print(f'D2H copy: {sum(runtime8)/len(runtime8)*1000:.4f}ms')
print(f'CHW > HWC: {sum(runtime9)/len(runtime9)*1000:.4f}ms')
print(f'Display: {sum(runtime0)/len(runtime0)*1000:.4f}ms')
print(f'Total (FPS): {sum(runtimeT)/len(runtimeT)*1000:.4f}ms / {len(runtimeT)/sum(runtimeT):.4f}')
cam.release()
cv2.destroyAllWindows()
