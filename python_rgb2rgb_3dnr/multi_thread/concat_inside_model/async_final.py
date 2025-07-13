import cv2
import time
import asyncio
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import tensorrt as trt
import numpy as np
import sys

from config import *

class MyProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
    
    def report_layer_time(self, layer_name, ms):
        print(layer_name, ms)

#########################################################################
############################## set cam ##################################
#########################################################################
cam = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
#########################################################################
#########################################################################
#########################################################################

#########################################################################
########################## load engine ##################################
#########################################################################
ctx = cuda.Device(device_num).make_context()
stream = cuda.Stream()
trt_logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(trt_logger)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
context.profiler = MyProfiler()
#########################################################################
#########################################################################
#########################################################################

#########################################################################
########################## allocate buffers #############################
#########################################################################
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
#########################################################################
#########################################################################
#########################################################################

async def H2D_D2H():
    global pipeline_latency, total_frames, host_inputs, cuda_inputs, host_outputs, cuda_outputs
    i = 0
    passed = False
    d = display_input
    pipeline_start = time.time()
    while i < total_frames:
        #########################################################################
        ######################### task 3 start ##################################
        #########################################################################
        
        if not Q2.empty():
            # print(f"{i}th task 3 start!\t{(time.time() - start)*1000:.0f}")
            host_outputs, cuda_outputs = Q2.get_nowait()
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            output = host_outputs[0].reshape(outp_shape)

            output = np.moveaxis(output[0], 0, 2)
            pipeline_time = round((time.time() - pipeline_start) * 1000, 2)
            pipeline_latency.append(pipeline_time)
            # print(f'{i}th time: {pipeline_time:2f}ms')
            pipeline_start = time.time()
            if d:
                cv2.imshow('asyncio with 3 tasks', noisy_frame_orig)
            else:
                cv2.imshow('asyncio with 3 tasks', output)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
            elif cv2.waitKey(1) & 0xFF in [ord('c')]:
                d = not d
            # print(f"{i}th task 3 end!  \t{(time.time() - start)*1000:.0f}")
        #########################################################################
        ######################### task 3 end ####################################
        #########################################################################
        
        #########################################################################
        ######################### task 1 start ##################################
        #########################################################################
        if not Q1.full():
            # print(f"{i}th task 1 start!\t{(time.time() - start)*1000:.0f}")
            _, noisy_frame_orig = cam.read()
            noisy_frame = np.expand_dims(np.moveaxis(noisy_frame_orig, 2, 0), axis=0)
            noisy_frame = np.ascontiguousarray(noisy_frame).ravel().astype(np.float32)
            if not passed:
                for j in range(frame_cnt):
                    np.copyto(host_inputs[j], noisy_frame)
                    cuda.memcpy_htod_async(cuda_inputs[j], host_inputs[j], stream)
                passed = True
            else:
                np.copyto(host_inputs[-1], noisy_frame)
                cuda.memcpy_htod_async(cuda_inputs[-1], host_inputs[-1], stream)
            
            Q1.put_nowait((host_inputs, cuda_inputs))
          
            # print(f"{i}th task 1 end!  \t{(time.time() - start)*1000:.0f}")
            i += 1
        #########################################################################
        ######################### task 1 end ####################################
        #########################################################################
        await asyncio.sleep(0)
        

async def Model():
    global model_latency, total_frames, bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs
    i = 0
    queue_start = None
    while i < total_frames:
        #########################################################################
        ######################### task 2 start ##################################
        #########################################################################
        if not Q1.empty():
            # print(f"{i}th task 2 start!\t{(time.time() - start)*1000:.0f}")
            host_inputs, cuda_inputs = Q1.get_nowait()
            model_start = time.time()
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # stream.synchronize()
            model_time = round((time.time() - model_start) * 1000, 2)
            print(f'{i}th model {model_time}ms')
            model_latency.append(model_time)
            host_inputs.append(host_inputs[0])
            host_inputs = host_inputs[1:]
            cuda_inputs.append(cuda_inputs[0])
            cuda_inputs = cuda_inputs[1:]
            bindings.insert(frame_cnt, bindings[0])
            bindings = bindings[1:]
            
            await asyncio.sleep(0.025)
            if not Q2.full():
           	    Q2.put_nowait((host_outputs, cuda_outputs))
            
            # print(f"{i}th task 2 end!  \t{(time.time() - start)*1000:.0f}")
            i += 1
        else:
            await asyncio.sleep(0.001)
        #########################################################################
        ######################### task 2 end ####################################
        #########################################################################

async def main():
    task1 = asyncio.create_task(H2D_D2H())
    task2 = asyncio.create_task(Model())
    await task1
    await task2

if __name__=='__main__':
    model_latency = []
    pipeline_latency = []
    Q1 = asyncio.Queue(maxsize=queue_size)
    Q2 = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_event_loop()
    start = time.time()
    loop.run_until_complete(main())
    runtime = time.time() - start
    print(f"Model latency: {sum(model_latency)/len(model_latency):.2f}ms")
    print(f"Pipeline latency: {sum(pipeline_latency)/len(pipeline_latency):.2f}ms")
    loop.close()
    cam.release()
    cv2.destroyAllWindows()
